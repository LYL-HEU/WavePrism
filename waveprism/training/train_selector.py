from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data.hdf5 import DayBlockSampler, GlobalH5WaveDataset, inspect_data
from ..models.expert_selector import FrozenStage1WindEncoder, Stage2Config
from ..models.factory import set_stage1_context
from ..models.probabilistic_forecaster import SelectorProbabilityModel
from .common import (
    PointMetricTracker,
    ProbMetricTracker,
    _init_metric_agg,
    _reduce_and_finalize_agg,
    _update_metric_agg,
    atomic_save,
    barrier,
    cleanup_distributed,
    count_parameters,
    ensure_dir,
    format_array,
    is_rank0,
    move_batch_to_device,
    set_requires_grad,
    set_seed,
    setup_distributed,
    tensor_to_float,
    torch_load_safe,
    unwrap_model,
)


def update_config_from_data(cfg: Stage2Config) -> Tuple[Stage2Config, list]:
    if not os.path.exists(cfg.data_h5):
        raise FileNotFoundError(cfg.data_h5)

    with h5py.File(cfg.data_h5, "r") as f:
        for key in [
            "wind",
            "satellite",
            "target_mask",
            "train_indices",
            "val_indices",
            "test_indices",
        ]:
            if key not in f:
                raise KeyError(f"HDF5 missing key: {key}")

        wind_shape = tuple(f["wind"].shape)
        # Number/order of experts come from Stage 1, not the numerical wave archive.
        pretrained = torch_load_safe(cfg.stage1_checkpoint, map_location="cpu")
        meta = pretrained["architecture"]
        wave_shape = (wind_shape[0], int(meta["num_experts"]), *wind_shape[-2:])
        sat_shape = tuple(f["satellite"].shape)

        if len(wind_shape) != 4:
            raise ValueError(f"wind should be [N,C,H,W], got {wind_shape}")
        if len(sat_shape) != 3:
            raise ValueError(f"satellite should be [N,H,W], got {sat_shape}")

        n, c, h, w = wind_shape
        nw, k, hw, ww = wave_shape
        ns, hs, ws = sat_shape
        if not (n == nw == ns and h == hw == hs and w == ww == ws):
            raise ValueError(
                f"HDF5 grid mismatch: wind={wind_shape}, wave={wave_shape}, satellite={sat_shape}"
            )

        cfg.in_time = int(meta["history_hours"])
        cfg.in_channels = int(c)
        cfg.num_experts = int(k)

        if cfg.expected_num_experts is not None and k != int(cfg.expected_num_experts):
            raise ValueError(
                f"Expected {cfg.expected_num_experts} experts, but Stage-1 checkpoint has K={k}."
            )
        if cfg.expected_h is not None and h != int(cfg.expected_h):
            raise ValueError(f"Expected H={cfg.expected_h}, got H={h}.")
        if cfg.expected_w is not None and w != int(cfg.expected_w):
            raise ValueError(f"Expected W={cfg.expected_w}, got W={w}.")
        if cfg.expected_time_step_hours is not None:
            h5_dt = f.attrs.get("time_step_hours", None)
            if h5_dt is None:
                raise KeyError("HDF5 missing attribute 'time_step_hours'.")
            if int(h5_dt) != int(cfg.expected_time_step_hours):
                raise ValueError(
                    f"Expected time_step_hours={cfg.expected_time_step_hours}, got {h5_dt}."
                )

        expert_names = list(pretrained["expert_names"])

        n_train = int(f["train_indices"].shape[0])
        n_val = int(f["val_indices"].shape[0])
        n_test = int(f["test_indices"].shape[0])

    if cfg.use_static_features:
        if not cfg.static_npz or not os.path.exists(cfg.static_npz):
            raise FileNotFoundError(
                f"Static features enabled but static NPZ not found: \n{cfg.static_npz}"
            )
        sdata = np.load(cfg.static_npz)
        if "static" not in sdata:
            raise KeyError(f"{cfg.static_npz} must contain 'static'.")
        static_shape = tuple(sdata["static"].shape)
        if len(static_shape) != 3:
            raise ValueError(f"static should be [S,H,W], got {static_shape}")
        if static_shape[-2:] != (h, w):
            raise ValueError(
                f"static H/W {static_shape[-2:]} != HDF5 H/W {(h, w)}.\n"
                "Regenerate static features using HDF5 lat/lon coordinates."
            )
        cfg.static_channels = int(static_shape[0])
    else:
        cfg.static_channels = 0

    # Validate TRAIN-only spatial expert skill prior

    # Spatial prior is derived from generated experts after loading Stage 1.
    print("\nConfig updated from global HDF5:")
    print(f"  wind shape     = {wind_shape}")
    print(f"  generated wave = {wave_shape} (Stage-1 decoder output)")
    print(f"  satellite      = {sat_shape}")
    print("  x_in used      = full chronological history; length taken from Stage-1 checkpoint")
    print(f"  num_experts    = {cfg.num_experts}")
    print(f"  expert_names   = {expert_names}")
    print(f"  train/val/test = {n_train}/{n_val}/{n_test}")
    if cfg.use_static_features:
        print(f"  static shape   = {static_shape}")
    if cfg.use_expert_skill_prior:
        print("  skill prior    = computed/cached from generated experts and TRAIN observations")

    return cfg, expert_names


_STAGE1_CONTEXT = None


def prepare_stage1_context(cfg, expert_names):
    global _STAGE1_CONTEXT
    checkpoint = torch_load_safe(cfg.stage1_checkpoint, map_location="cpu")
    required = {"architecture", "model", "mean", "std", "ocean_mask", "expert_names"}
    if not required.issubset(checkpoint):
        raise ValueError("Expected a WavePrism Stage-1 v2 training checkpoint (best.pt).")
    meta = dict(checkpoint["architecture"])
    if meta.get("version") != 2:
        raise ValueError("Only the supplied Stage-1 v2 encoder architecture is supported.")
    args = SimpleNamespace(
        data_h5=cfg.data_h5,
        static_npz=cfg.static_npz,
        expected_experts=cfg.num_experts,
        ocean_mask_key="",
        history_hours=meta["history_hours"],
        latent_channels=meta["latent_channels"],
        decoder_channels=meta["decoder_channels"],
        temporal_channels=meta["temporal_channels"],
        dropout=meta["dropout"],
        no_decoder_checkpoint=not meta["checkpoint_decoders"],
    )
    actual, splits, valid, geometry, ocean, names = inspect_data(
        args, ocean_override=checkpoint["ocean_mask"]
    )
    if actual != meta or names != list(checkpoint["expert_names"]) or names != expert_names:
        raise ValueError("Stage-1 architecture or expert order does not match this dataset.")
    # When the original source is still accessible, verify its grid/time/splits.
    # If it has moved, the caller must supply the identical data/static copies.
    old_path = Path(checkpoint.get("data_h5", ""))
    if old_path.is_file() and old_path.resolve() != Path(cfg.data_h5).resolve():
        with h5py.File(old_path, "r") as old, h5py.File(cfg.data_h5, "r") as current:
            for key in ("time", "lat", "lon", "train_indices", "val_indices", "test_indices"):
                if key not in old or not np.array_equal(old[key][:], current[key][:]):
                    raise ValueError("Stage-1 source differs in " + key)
    if "stage1_architecture_json" in checkpoint:
        raise ValueError("Expected training checkpoint, not export metadata.")
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    if (
        mean.shape != (cfg.in_channels,)
        or std.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError("Invalid Stage-1 wind normalization statistics.")
    if not cfg.normalize_input:
        raise ValueError("The frozen Stage-1 encoder requires its training normalization.")
    if not cfg.use_static_features:
        raise ValueError("This pretrained Stage-1 encoder requires the original static NPZ.")
    cfg.in_time = int(meta["history_hours"])
    cfg.wind_base_channels = int(meta["latent_channels"])
    cfg.stage1_architecture = meta
    digest = hashlib.sha256()
    with open(cfg.stage1_checkpoint, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    cfg.stage1_checkpoint_sha256 = digest.hexdigest()
    _STAGE1_CONTEXT = dict(
        checkpoint=checkpoint,
        meta=meta,
        splits=splits,
        valid=valid,
        geometry=geometry,
        ocean=ocean,
        mean=mean,
        std=std,
    )
    set_stage1_context(_STAGE1_CONTEXT)
    print(
        f"Frozen Stage-1 encoder: history={cfg.in_time} hours, z={cfg.wind_base_channels} channels"
    )
    print(f"Wind normalization loaded from checkpoint: mean={mean}, std={std}")
    print("Expert source: all Stage-1 decoder predictions (frozen); HDF5 wave values are not read.")


@torch.no_grad()
def prepare_generated_batch(model, batch):
    raw = unwrap_model(model)
    z, experts = raw.selector.wind_encoder(batch["x_in"])
    result = dict(batch)
    result["x_in"] = z
    result["m_mod"] = experts
    result["mask"] = result["mask"] * torch.isfinite(experts).any(1, keepdim=True)
    return result


def compute_or_load_wind_stats(cfg, rank):
    # No rescan and no second normalization: the checkpoint is authoritative.
    return _STAGE1_CONTEXT["mean"].copy(), _STAGE1_CONTEXT["std"].copy()


def compute_or_load_generated_statistics(
    cfg, expert_names, rank, world_size, device, backbone, train_dataset
):
    """Exact disjoint TRAIN samples on all ranks; no WW3 wave reads.

    Per-expert MAE maps define logits -(MAE-min_MAE)/temperature; confidence
    is observation support times the best/second-best MAE gap. Unobserved cells
    get zero logits/confidence (no preference). Class weighting retains the
    original sqrt_inv_freq formula, now for generated-expert oracle labels.
    """
    ensure_dir(cfg.save_dir)
    cache_path = Path(cfg.save_dir) / "stage1_generated_train_statistics.npz"
    cfg.expert_skill_prior_npz = str(cache_path)
    with h5py.File(cfg.data_h5, "r") as f:
        lat, lon = np.asarray(f["lat"][:]), np.asarray(f["lon"][:])
    digest = hashlib.sha256()
    for a in (train_dataset.indices, train_dataset.times, lat, lon, _STAGE1_CONTEXT["geometry"]):
        digest.update(np.ascontiguousarray(a).tobytes())
    source_stat = Path(cfg.data_h5).stat()
    signature = json.dumps(
        dict(
            version=1,
            checkpoint=cfg.stage1_checkpoint_sha256,
            source=str(Path(cfg.data_h5).resolve()),
            size=source_stat.st_size,
            mtime_ns=source_stat.st_mtime_ns,
            arrays=digest.hexdigest(),
            encoder_amp=cfg.encoder_amp,
            class_mode=cfg.class_weight_mode,
            class_min=cfg.class_weight_min,
            class_max=cfg.class_weight_max,
            hs_threshold=cfg.train_hs_threshold,
            margin=cfg.use_margin_label,
            margin_m=cfg.hard_label_margin,
            prior_temperature=cfg.generated_prior_temperature_m,
            prior_count_scale=cfg.generated_prior_count_scale,
        ),
        sort_keys=True,
    )
    reuse = False
    if rank == 0 and not cfg.recompute_class_weights and cache_path.is_file():
        try:
            with np.load(cache_path) as c:
                reuse = (
                    str(c["signature"].item()) == signature
                    and c["weights"].shape == (cfg.num_experts,)
                    and c["prior_logits"].shape == (cfg.num_experts, *train_dataset.ocean.shape)
                    and c["prior_confidence"].shape == (1, *train_dataset.ocean.shape)
                )
        except (OSError, ValueError, KeyError):
            reuse = False
    flag = torch.tensor([int(reuse)], dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.broadcast(flag, src=0)
    if not flag.item():
        if rank == 0:
            print(
                "Computing TRAIN-only prior/class weights from GENERATED Stage-1 experts; first run only.",
                flush=True,
            )
        h, w = train_dataset.ocean.shape
        sums = torch.zeros(cfg.num_experts, h, w, dtype=torch.float64, device=device)
        counts = torch.zeros_like(sums)
        oracle_counts = torch.zeros(cfg.num_experts, dtype=torch.float64, device=device)
        subset = torch.utils.data.Subset(train_dataset, range(rank, len(train_dataset), world_size))
        batches = DataLoader(
            subset,
            batch_size=cfg.batch_size_per_gpu,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
        )
        for batch in tqdm(batches, desc="Generated-expert TRAIN statistics", disable=rank != 0):
            batch = move_batch_to_device(batch, device)
            _, experts = backbone(batch["x_in"])
            y = batch["y_sat"]
            valid = (batch["mask"] > 0.5) & torch.isfinite(y) & torch.isfinite(experts)
            if cfg.train_hs_threshold is not None:
                valid &= y >= float(cfg.train_hs_threshold)
            error = torch.where(valid, (experts - y).abs(), torch.zeros_like(experts))
            sums += error.double().sum(0)
            counts += valid.double().sum(0)
            label, _, _ = build_oracle_label(
                experts,
                y,
                batch["mask"],
                hs_threshold=cfg.train_hs_threshold,
                use_margin_label=cfg.use_margin_label,
                hard_label_margin=cfg.hard_label_margin,
            )
            labels = label[label >= 0]
            oracle_counts += torch.bincount(labels, minlength=cfg.num_experts).double()
        if dist.is_initialized():
            for value in (sums, counts, oracle_counts):
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if not counts.sum().item() or not oracle_counts.sum().item():
            raise ValueError("No valid TRAIN observations for generated-expert statistics.")
        if rank == 0:
            mae = (sums / counts.clamp_min(1)).cpu().numpy()
            support = counts.cpu().numpy()
            valid = support > 0
            sortable = np.where(valid, mae, np.inf)
            best = sortable.min(0, keepdims=True)
            observed = np.isfinite(best)
            # Avoid inf arithmetic in cells with no observations.
            best_finite = np.where(observed, best, 0.0)
            logits = np.where(
                valid, -(mae - best_finite) / cfg.generated_prior_temperature_m, -20.0
            )
            logits = np.where(observed, np.clip(logits, -20.0, 0.0), 0.0).astype("f4")
            if cfg.num_experts > 1:
                second = np.partition(sortable, 1, axis=0)[1:2]
                gap = np.where(np.isfinite(second), second, best_finite) - best_finite
            else:
                gap = np.zeros_like(best_finite)
            nobs = support.max(0, keepdims=True)
            confidence = (
                nobs
                / (nobs + cfg.generated_prior_count_scale)
                * np.clip(gap / cfg.generated_prior_temperature_m, 0.0, 1.0)
            )
            confidence = np.where(observed, confidence, 0.0).astype("f4")
            raw_counts = oracle_counts.cpu().numpy()
            safe_counts = np.maximum(raw_counts, 1.0)
            freq = safe_counts / safe_counts.sum()
            if cfg.class_weight_mode == "none":
                weights = np.ones_like(freq)
            elif cfg.class_weight_mode == "inv_freq":
                weights = 1.0 / freq
            elif cfg.class_weight_mode == "sqrt_inv_freq":
                weights = np.sqrt(1.0 / freq)
            else:
                raise ValueError("Unsupported class_weight_mode: " + cfg.class_weight_mode)
            weights = np.clip(
                weights / weights.mean(), cfg.class_weight_min, cfg.class_weight_max
            ).astype("f4")
            # Keep unsupported mean errors identifiable in the saved diagnostics.
            mae = np.where(valid, mae, np.nan).astype("f4")
            temporary = str(cache_path) + ".tmp"
            with open(temporary, "wb") as stream:
                np.savez(
                    stream,
                    signature=signature,
                    prior_logits=logits,
                    prior_confidence=confidence,
                    expert_names=np.asarray(expert_names),
                    lat=lat,
                    lon=lon,
                    build_split="train",
                    expert_source=cfg.expert_source,
                    stage1_checkpoint_sha256=cfg.stage1_checkpoint_sha256,
                    counts=raw_counts,
                    freq=freq,
                    weights=weights,
                    mae=mae,
                    observation_count=support.astype("i8"),
                )
            os.replace(temporary, cache_path)
            print("Saved generated-expert TRAIN statistics:", cache_path)
    elif rank == 0:
        print("Reusing generated-expert TRAIN statistics:", cache_path)
    barrier()
    with np.load(cache_path) as cached:
        weights = cached["weights"].astype("f4")
    return weights


def build_datasets_and_loaders(
    cfg: Stage2Config,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    rank: int,
    world_size: int,
    distributed: bool,
):
    static_npz = cfg.static_npz if cfg.use_static_features else None

    train_ds = GlobalH5WaveDataset(
        cfg.data_h5,
        "train",
        cfg.normalize_input,
        input_mean,
        input_std,
        static_npz,
        cfg.use_position_features,
    )
    val_ds = GlobalH5WaveDataset(
        cfg.data_h5,
        "val",
        cfg.normalize_input,
        input_mean,
        input_std,
        static_npz,
        cfg.use_position_features,
    )
    test_ds = GlobalH5WaveDataset(
        cfg.data_h5,
        "test",
        cfg.normalize_input,
        input_mean,
        input_std,
        static_npz,
        cfg.use_position_features,
    )

    # Nearby histories reuse the raw input window instead of rereading 241 hours.
    train_sampler = DayBlockSampler(train_ds, world_size, rank, cfg.batch_size_per_gpu, cfg.seed)
    train_shuffle = False
    if rank == 0:
        print(
            f"Training drops {len(train_ds) - train_sampler.total} targets per epoch to avoid padded duplicates."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size_per_gpu,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size_per_gpu,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size_per_gpu,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, train_sampler


def build_oracle_label(
    m_mod: torch.Tensor,
    y_sat: torch.Tensor,
    mask: torch.Tensor,
    hs_threshold: Optional[float] = None,
    use_margin_label: bool = False,
    hard_label_margin: float = 0.05,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    label = argmin_k |m_mod_k - y_sat|
    """
    expert_valid = torch.isfinite(m_mod)
    expert_count = expert_valid.sum(dim=1, keepdim=True)

    valid = (mask > 0.5) & torch.isfinite(y_sat) & (expert_count >= 1)

    if hs_threshold is not None:
        valid = valid & (y_sat >= hs_threshold)

    m_safe = torch.nan_to_num(m_mod, nan=0.0, posinf=0.0, neginf=0.0)

    abs_err = torch.abs(m_safe - y_sat)
    abs_err = torch.where(expert_valid, abs_err, torch.full_like(abs_err, 1.0e6))

    best_err, best_idx = torch.min(abs_err, dim=1, keepdim=True)

    if use_margin_label:
        valid_for_top2 = valid & (expert_count >= 2)
        top2_err, _ = torch.topk(abs_err, k=2, dim=1, largest=False)
        second_err = top2_err[:, 1:2, :, :]
        confident = (second_err - best_err) >= hard_label_margin
        valid = valid_for_top2 & confident

    label = best_idx[:, 0, :, :].long()
    ignore = torch.full_like(label, ignore_index)
    label = torch.where(valid[:, 0, :, :], label, ignore)

    aux = {"best_err": best_err, "expert_valid": expert_valid.float(), "m_safe": m_safe}

    return label, valid.float(), aux


def build_soft_oracle_target(
    m_mod: torch.Tensor,
    y_sat: torch.Tensor,
    mask: torch.Tensor,
    hs_threshold: Optional[float] = None,
    temperature: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Build soft oracle probabilities from expert absolute errors; tau is in metres."""
    if temperature <= 0:
        raise ValueError("soft_oracle_temperature must be > 0")

    expert_valid = torch.isfinite(m_mod)
    expert_any = expert_valid.any(dim=1, keepdim=True)
    valid = (mask > 0.5) & torch.isfinite(y_sat) & expert_any
    if hs_threshold is not None:
        valid = valid & (y_sat >= hs_threshold)

    m_safe = torch.nan_to_num(m_mod, nan=0.0, posinf=0.0, neginf=0.0)
    abs_err = torch.abs(m_safe - y_sat)
    abs_err_masked = torch.where(expert_valid, abs_err, torch.full_like(abs_err, 1.0e6))
    best_err = abs_err_masked.min(dim=1, keepdim=True).values
    relative_err = abs_err_masked - best_err

    target_logits = -relative_err / float(temperature)
    target_logits = target_logits.masked_fill(~expert_valid, -1.0e4)
    q = F.softmax(target_logits, dim=1)
    q = q * valid.float()

    return (
        q,
        valid.float(),
        {"abs_err": abs_err_masked, "best_err": best_err, "relative_err": relative_err},
    )


def weighted_soft_cross_entropy(
    logits: torch.Tensor,
    soft_target: torch.Tensor,
    valid: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute pixelwise soft-label cross entropy with optional class weights."""
    log_prob = F.log_softmax(logits, dim=1)
    target = soft_target

    if class_weights is not None:
        w = class_weights.view(1, -1, 1, 1).to(logits.dtype)
        weighted_target = target * w
        denom = weighted_target.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weighted_target = weighted_target / denom
    else:
        weighted_target = target

    per_pixel = -(weighted_target * log_prob).sum(dim=1, keepdim=True)
    valid_f = valid.float()
    return (per_pixel * valid_f).sum() / valid_f.sum().clamp_min(1.0)


def make_valid_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    hs_threshold: Optional[float] = None,
) -> torch.Tensor:
    valid = (mask > 0.5) & torch.isfinite(pred) & torch.isfinite(target)

    if hs_threshold is not None:
        valid = valid & (target >= hs_threshold)

    return valid.float()


def pairwise_regret_ranking_loss(
    logits_masked: torch.Tensor,
    m_mod: torch.Tensor,
    y_sat: torch.Tensor,
    mask: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    hs_threshold: Optional[float] = None,
    gap_scale_m: float = 0.10,
    min_gap_m: float = 1e-6,
    use_class_weights: bool = True,
    pair_chunk_size: int = 24,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Rank expert pairs by absolute error, weighting each pair by its error gap."""
    if gap_scale_m <= 0:
        raise ValueError("gap_scale_m must be > 0.")
    if pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be > 0.")

    _, K, _, _ = logits_masked.shape

    expert_valid = torch.isfinite(m_mod)
    target_valid = (mask > 0.5) & torch.isfinite(y_sat)
    if hs_threshold is not None:
        target_valid = target_valid & (y_sat >= hs_threshold)

    m_safe = torch.nan_to_num(m_mod, nan=0.0, posinf=0.0, neginf=0.0)
    y_safe = torch.nan_to_num(y_sat, nan=0.0, posinf=0.0, neginf=0.0)
    abs_err = torch.abs(m_safe - y_safe)

    # [2,P], P=K*(K-1)/2; for K=15, P=105.
    pair_index = torch.triu_indices(K, K, offset=1, device=logits_masked.device)
    pair_i_all = pair_index[0]
    pair_j_all = pair_index[1]
    num_pairs = int(pair_i_all.numel())

    zero = logits_masked.sum() * 0.0
    numerator = zero
    denominator = zero

    # Diagnostics do not require gradients.
    acc_numerator = zero.detach().clone()
    mean_gap_numerator = zero.detach().clone()
    pair_count = zero.detach().clone()

    target_valid_2d = target_valid[:, 0, :, :]

    # class_weights is tiny [K], keep on same device/dtype.
    cw = None
    if use_class_weights and class_weights is not None:
        cw = class_weights.to(device=logits_masked.device, dtype=logits_masked.dtype)

    for start_pair in range(0, num_pairs, int(pair_chunk_size)):
        end_pair = min(start_pair + int(pair_chunk_size), num_pairs)
        pair_i = pair_i_all[start_pair:end_pair]
        pair_j = pair_j_all[start_pair:end_pair]
        p = int(pair_i.numel())

        # Advanced indexing only on the small expert dimension:
        # [B,P,H,W], P<=pair_chunk_size.
        ei = abs_err[:, pair_i, :, :]
        ej = abs_err[:, pair_j, :, :]
        si = logits_masked[:, pair_i, :, :]
        sj = logits_masked[:, pair_j, :, :]

        pair_valid = (
            target_valid_2d[:, None, :, :]
            & expert_valid[:, pair_i, :, :]
            & expert_valid[:, pair_j, :, :]
        )

        gap_signed = ej - ei
        gap_abs = torch.abs(gap_signed)
        pair_valid = pair_valid & (gap_abs > float(min_gap_m))

        # +1 => i better -> require si>sj; -1 => j better -> require sj>si.
        direction = torch.where(
            gap_signed >= 0.0, torch.ones_like(gap_signed), -torch.ones_like(gap_signed)
        )
        score_margin = direction * (si - sj)
        pair_loss = F.softplus(-score_margin)

        regret_weight = torch.clamp(gap_abs / float(gap_scale_m), min=0.0, max=1.0)

        if cw is not None:
            cw_i = cw[pair_i].view(1, p, 1, 1)
            cw_j = cw[pair_j].view(1, p, 1, 1)
            better_class_weight = torch.where(gap_signed >= 0.0, cw_i, cw_j)
            pair_weight = regret_weight * better_class_weight
        else:
            pair_weight = regret_weight

        valid_f = pair_valid.to(dtype=logits_masked.dtype)
        w = pair_weight * valid_f

        numerator = numerator + (pair_loss * w).sum()
        denominator = denominator + w.sum()

        with torch.no_grad():
            ordered_correct = (score_margin > 0.0).to(logits_masked.dtype)
            acc_numerator = acc_numerator + (ordered_correct * w).sum()
            mean_gap_numerator = mean_gap_numerator + (gap_abs * valid_f).sum()
            pair_count = pair_count + valid_f.sum()

    loss = numerator / denominator.clamp_min(1.0)

    with torch.no_grad():
        weighted_pair_acc = acc_numerator / denominator.detach().clamp_min(1.0)
        mean_pair_gap = mean_gap_numerator / pair_count.clamp_min(1.0)

    aux = {
        "weighted_pair_acc": weighted_pair_acc,
        "mean_pair_gap_m": mean_pair_gap,
        "pair_count": pair_count,
        "num_expert_pairs": torch.tensor(
            float(num_pairs), dtype=logits_masked.dtype, device=logits_masked.device
        ),
    }
    return loss, aux


def straight_through_hard_selection_cost(
    logits_masked: torch.Tensor,
    m_mod: torch.Tensor,
    y_sat: torch.Tensor,
    mask: torch.Tensor,
    hs_threshold: Optional[float] = None,
    temperature: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Use argmax cost in the forward pass and softmax gradients in the backward pass."""
    if temperature <= 0:
        raise ValueError("st_temperature must be > 0.")

    expert_valid = torch.isfinite(m_mod)
    expert_any = expert_valid.any(dim=1, keepdim=True)

    valid = (mask > 0.5) & torch.isfinite(y_sat) & expert_any
    if hs_threshold is not None:
        valid = valid & (y_sat >= hs_threshold)

    m_safe = torch.nan_to_num(m_mod, nan=0.0, posinf=0.0, neginf=0.0)
    y_safe = torch.nan_to_num(y_sat, nan=0.0, posinf=0.0, neginf=0.0)

    abs_err = torch.abs(m_safe - y_safe)
    abs_err = torch.where(expert_valid, abs_err, torch.zeros_like(abs_err))

    # backward surrogate
    p_soft = F.softmax(logits_masked / float(temperature), dim=1)

    hard_idx = torch.argmax(logits_masked, dim=1, keepdim=True)

    p_hard = (
        F.one_hot(hard_idx[:, 0, :, :].long(), num_classes=logits_masked.shape[1])
        .permute(0, 3, 1, 2)
        .to(dtype=logits_masked.dtype)
    )

    p_st = p_hard - p_soft.detach() + p_soft

    hard_error_map = torch.sum(p_st * abs_err, dim=1, keepdim=True)

    valid_f = valid.float()
    loss = (hard_error_map * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    with torch.no_grad():
        selected_error_map = torch.sum(p_hard * abs_err, dim=1, keepdim=True)

        inf = torch.full_like(abs_err, float("inf"))
        err_for_min = torch.where(expert_valid, abs_err, inf)
        oracle_error_map = err_for_min.min(dim=1, keepdim=True).values
        oracle_error_map = torch.where(
            torch.isfinite(oracle_error_map), oracle_error_map, torch.zeros_like(oracle_error_map)
        )

        hard_mae = (selected_error_map * valid_f).sum() / valid_f.sum().clamp_min(1.0)
        oracle_mae = (oracle_error_map * valid_f).sum() / valid_f.sum().clamp_min(1.0)
        mean_regret = hard_mae - oracle_mae

        soft_expected_error_map = torch.sum(p_soft * abs_err, dim=1, keepdim=True)
        soft_expected_mae = (soft_expected_error_map * valid_f).sum() / valid_f.sum().clamp_min(1.0)

        mean_conf = (
            p_soft.max(dim=1, keepdim=True).values * valid_f
        ).sum() / valid_f.sum().clamp_min(1.0)

    aux = {
        "hard_mae": hard_mae,
        "oracle_mae": oracle_mae,
        "mean_regret": mean_regret,
        "soft_expected_mae": soft_expected_mae,
        "mean_confidence": mean_conf,
        "valid_pixels": valid_f.sum(),
    }

    return loss, aux


def selector_loss_fn(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: Stage2Config,
    class_weights: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Combine pairwise ranking, straight-through selection cost, and soft oracle supervision."""
    logits = outputs["logits_masked"]
    m_mod = batch["m_mod"]
    y_sat = batch["y_sat"]
    mask = batch["mask"]

    label, hard_valid, _ = build_oracle_label(
        m_mod=m_mod,
        y_sat=y_sat,
        mask=mask,
        hs_threshold=cfg.train_hs_threshold,
        use_margin_label=False,
        hard_label_margin=0.0,
        ignore_index=-100,
    )

    # 1) Pairwise regret-aware ranking
    loss_rank, rank_aux = pairwise_regret_ranking_loss(
        logits_masked=logits,
        m_mod=m_mod,
        y_sat=y_sat,
        mask=mask,
        class_weights=class_weights,
        hs_threshold=cfg.train_hs_threshold,
        gap_scale_m=cfg.rank_gap_scale_m,
        min_gap_m=cfg.rank_min_gap_m,
        use_class_weights=cfg.ranking_use_class_weights,
        pair_chunk_size=cfg.rank_pair_chunk_size,
    )

    # 2) Straight-Through hard argmax cost
    loss_st, st_aux = straight_through_hard_selection_cost(
        logits_masked=logits,
        m_mod=m_mod,
        y_sat=y_sat,
        mask=mask,
        hs_threshold=cfg.train_hs_threshold,
        temperature=cfg.st_temperature,
    )

    # 3) Auxiliary Soft Oracle
    soft_target, soft_valid, _ = build_soft_oracle_target(
        m_mod=m_mod,
        y_sat=y_sat,
        mask=mask,
        hs_threshold=cfg.train_hs_threshold,
        temperature=cfg.soft_oracle_temperature,
    )

    if cfg.use_soft_oracle and cfg.lambda_soft_oracle > 0:
        loss_soft = weighted_soft_cross_entropy(
            logits=logits, soft_target=soft_target, valid=soft_valid, class_weights=class_weights
        )
    else:
        loss_soft = logits.sum() * 0.0

    total = (
        cfg.lambda_rank * loss_rank
        + cfg.lambda_st_hardcost * loss_st
        + cfg.lambda_soft_oracle * loss_soft
    )

    # Diagnostics
    with torch.no_grad():
        pred_idx = outputs["pred_idx"][:, 0, :, :]
        valid_flat = label >= 0

        if valid_flat.any():
            correct = (pred_idx[valid_flat] == label[valid_flat]).float().mean()
        else:
            correct = torch.tensor(0.0, device=logits.device)

        pred_freq = torch.zeros(cfg.num_experts, device=logits.device)
        label_freq = torch.zeros(cfg.num_experts, device=logits.device)

        if valid_flat.any():
            pred_valid = pred_idx[valid_flat]
            label_valid = label[valid_flat]

            pred_count = torch.bincount(pred_valid.reshape(-1), minlength=cfg.num_experts).float()
            label_count = torch.bincount(label_valid.reshape(-1), minlength=cfg.num_experts).float()

            pred_freq = pred_count / pred_count.sum().clamp_min(1.0)
            label_freq = label_count / label_count.sum().clamp_min(1.0)

        q = soft_target.clamp_min(1e-12)
        soft_entropy_map = -(q * torch.log(q)).sum(dim=1, keepdim=True)

        soft_entropy = (soft_entropy_map * soft_valid).sum() / soft_valid.sum().clamp_min(1.0)

    metrics = {
        "loss_total": tensor_to_float(total),
        "loss_rank": tensor_to_float(loss_rank),
        "loss_st_hardcost": tensor_to_float(loss_st),
        "loss_soft_oracle": tensor_to_float(loss_soft),
        "hard_selected_mae": tensor_to_float(st_aux["hard_mae"]),
        "oracle_mae": tensor_to_float(st_aux["oracle_mae"]),
        "mean_regret": tensor_to_float(st_aux["mean_regret"]),
        "soft_expected_mae": tensor_to_float(st_aux["soft_expected_mae"]),
        "st_mean_confidence": tensor_to_float(st_aux["mean_confidence"]),
        "pairwise_order_acc": tensor_to_float(rank_aux["weighted_pair_acc"]),
        "mean_pair_gap_m": tensor_to_float(rank_aux["mean_pair_gap_m"]),
        "acc": tensor_to_float(correct),
        "soft_target_entropy": tensor_to_float(soft_entropy),
        "pred_freq": (pred_freq.detach().cpu().numpy().tolist()),
        "label_freq": (label_freq.detach().cpu().numpy().tolist()),
        "valid_pixels": float(st_aux["valid_pixels"].detach().cpu().item()),
        "soft_valid_pixels": float(soft_valid.sum().detach().cpu().item()),
        # Spatial-prior diagnostics
        "skill_prior_alpha": (
            tensor_to_float(outputs["skill_prior_alpha"])
            if outputs.get("skill_prior_alpha") is not None
            else 0.0
        ),
        "skill_prior_confidence_mean": (
            tensor_to_float(outputs["skill_prior_confidence"].mean())
            if outputs.get("skill_prior_confidence") is not None
            else 0.0
        ),
    }

    return total, metrics


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    hs_threshold: Optional[float] = None,
) -> torch.Tensor:
    valid = make_valid_mask(pred, target, mask, hs_threshold)

    pred_safe = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target_safe = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    loss = torch.abs(pred_safe - target_safe) * valid
    return loss.sum() / valid.sum().clamp_min(1.0)


def masked_gaussian_nll(mu, sigma, target, mask, hs_threshold=None):
    """Evaluate NLL only at valid observations, before any division or logarithm."""
    valid = (
        (mask > 0.5)
        & torch.isfinite(target)
        & torch.isfinite(mu)
        & torch.isfinite(sigma)
        & (sigma > 0)
    )
    if hs_threshold is not None:
        valid &= target >= hs_threshold
    scale = sigma[valid].clamp_min(1e-6)
    residual = (target[valid] - mu[valid]) / scale
    nll = 0.5 * residual.square() + scale.log() + 0.5 * math.log(2.0 * math.pi)
    return nll.sum() / valid.sum().clamp_min(1)


def probability_loss_fn(
    outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: Stage2Config
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    mu = outputs["mu"]
    sigma = outputs["sigma"]

    y = batch["y_sat"]
    mask = batch["mask"]

    loss_nll = masked_gaussian_nll(
        mu=mu, sigma=sigma, target=y, mask=mask, hs_threshold=cfg.train_hs_threshold
    )

    loss_mu_mae = masked_l1_loss(pred=mu, target=y, mask=mask, hs_threshold=cfg.train_hs_threshold)

    total = cfg.lambda_nll * loss_nll + cfg.lambda_mu_mae * loss_mu_mae

    with torch.no_grad():
        valid = make_valid_mask(mu, y, mask, cfg.train_hs_threshold)
        sigma_valid = sigma[valid > 0.5]

        if sigma_valid.numel() > 0:
            sigma_mean = sigma_valid.mean()
            sigma_min = sigma_valid.min()
            sigma_max = sigma_valid.max()
        else:
            sigma_mean = torch.tensor(float("nan"), device=mu.device)
            sigma_min = torch.tensor(float("nan"), device=mu.device)
            sigma_max = torch.tensor(float("nan"), device=mu.device)

    metrics = {
        "loss_total": tensor_to_float(total),
        "loss_nll": tensor_to_float(loss_nll),
        "loss_mu_mae": tensor_to_float(loss_mu_mae),
        "sigma_mean": tensor_to_float(sigma_mean),
        "sigma_min": tensor_to_float(sigma_min),
        "sigma_max": tensor_to_float(sigma_max),
    }

    return total, metrics


@torch.no_grad()
def evaluate_selector(
    model: SelectorProbabilityModel, loader: DataLoader, cfg: Stage2Config
) -> Dict[str, Any]:
    model.eval()

    K = cfg.num_experts

    trackers = {
        "selector": PointMetricTracker(),
        "mean_ens": PointMetricTracker(),
        "hard_oracle": PointMetricTracker(),
        "convex_oracle": PointMetricTracker(),
    }

    for k in range(K):
        trackers[f"expert_{k:02d}"] = PointMetricTracker()

    selector_count = np.zeros(K, dtype=np.float64)
    oracle_count = np.zeros(K, dtype=np.float64)
    total_valid_for_freq = 0.0
    correct_count = 0.0
    label_count_total = 0.0

    for batch in tqdm(loader, desc="Selector metrics", dynamic_ncols=True, leave=False):
        batch = move_batch_to_device(batch, cfg.device)
        batch = prepare_generated_batch(model, batch)

        static = batch["static"] if "static" in batch else None
        position = batch["position"] if "position" in batch else None

        outputs = model.forward_selector(
            batch["x_in"], batch["m_mod"], static=static, position=position
        )

        y = batch["y_sat"]
        mask = batch["mask"]
        m_mod = batch["m_mod"]

        selected_pred = outputs["selected_pred"]
        pred_idx = outputs["pred_idx"]
        m_safe = outputs["m_safe"]
        expert_valid = outputs["expert_valid"]

        valid_base = (
            (mask > 0.5) & torch.isfinite(y) & torch.isfinite(m_mod).any(dim=1, keepdim=True)
        )

        if cfg.eval_hs_threshold is not None:
            valid_base = valid_base & (y >= cfg.eval_hs_threshold)

        valid_base = valid_base.float()

        trackers["selector"].update(selected_pred, y, valid_base)

        mean_ens = (m_safe * expert_valid).sum(dim=1, keepdim=True) / expert_valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        trackers["mean_ens"].update(mean_ens, y, valid_base)

        abs_err = torch.abs(m_safe - y)
        abs_err = torch.where(expert_valid > 0.5, abs_err, torch.full_like(abs_err, 1.0e6))

        best_idx = torch.argmin(abs_err, dim=1, keepdim=True)
        hard_oracle = torch.gather(m_safe, dim=1, index=best_idx)
        trackers["hard_oracle"].update(hard_oracle, y, valid_base)

        m_for_min = torch.where(expert_valid > 0.5, m_safe, torch.full_like(m_safe, float("inf")))
        m_for_max = torch.where(expert_valid > 0.5, m_safe, torch.full_like(m_safe, -float("inf")))
        m_min = torch.min(m_for_min, dim=1, keepdim=True).values
        m_max = torch.max(m_for_max, dim=1, keepdim=True).values
        m_min = torch.where(torch.isfinite(m_min), m_min, torch.zeros_like(m_min))
        m_max = torch.where(torch.isfinite(m_max), m_max, torch.zeros_like(m_max))

        convex_oracle = torch.minimum(torch.maximum(y, m_min), m_max)
        trackers["convex_oracle"].update(convex_oracle, y, valid_base)

        for k in range(K):
            pred_k = m_mod[:, k : k + 1, :, :]
            valid_k = valid_base * torch.isfinite(pred_k).float()
            trackers[f"expert_{k:02d}"].update(pred_k, y, valid_k)

        if valid_base.sum().item() > 0:
            total_valid_for_freq += valid_base.sum().item()

            for k in range(K):
                selector_count[k] += ((pred_idx == k).float() * valid_base).sum().item()
                oracle_count[k] += ((best_idx == k).float() * valid_base).sum().item()

            correct_count += ((pred_idx == best_idx).float() * valid_base).sum().item()
            label_count_total += valid_base.sum().item()

    result = {name: tracker.compute() for name, tracker in trackers.items()}

    if total_valid_for_freq > 0:
        result["selector_frequency"] = (selector_count / total_valid_for_freq).tolist()
        result["oracle_frequency"] = (oracle_count / total_valid_for_freq).tolist()
    else:
        result["selector_frequency"] = [float("nan")] * K
        result["oracle_frequency"] = [float("nan")] * K

    if label_count_total > 0:
        result["selector_vs_oracle_acc"] = float(correct_count / label_count_total)
    else:
        result["selector_vs_oracle_acc"] = float("nan")

    expert_maes = {name: result[name]["mae"] for name in result if name.startswith("expert_")}
    best_expert_name = min(expert_maes, key=expert_maes.get)

    result["best_expert_name"] = best_expert_name
    result["best_expert_mae"] = result[best_expert_name]["mae"]
    result["best_expert_rmse"] = result[best_expert_name]["rmse"]

    result["selector_improve_mae_vs_best_expert_percent"] = (
        (result["best_expert_mae"] - result["selector"]["mae"])
        / max(result["best_expert_mae"], 1e-12)
        * 100.0
    )

    result["selector_improve_rmse_vs_best_expert_percent"] = (
        (result["best_expert_rmse"] - result["selector"]["rmse"])
        / max(result["best_expert_rmse"], 1e-12)
        * 100.0
    )

    return result


@torch.no_grad()
def evaluate_probability(
    model: SelectorProbabilityModel, loader: DataLoader, cfg: Stage2Config
) -> Dict[str, Any]:
    model.eval()

    point_mu = PointMetricTracker()
    point_selected = PointMetricTracker()
    prob_tracker = ProbMetricTracker()

    for batch in tqdm(loader, desc="Probability metrics", dynamic_ncols=True, leave=False):
        batch = move_batch_to_device(batch, cfg.device)
        batch = prepare_generated_batch(model, batch)

        static = batch["static"] if "static" in batch else None
        position = batch["position"] if "position" in batch else None

        outputs = model.forward_probability(
            batch["x_in"], batch["m_mod"], static=static, position=position, detach_selector=True
        )

        y = batch["y_sat"]
        mask = batch["mask"]

        mu = outputs["mu"]
        sigma = outputs["sigma"]
        selected_pred = outputs["selected_pred"]

        valid = make_valid_mask(mu, y, mask, cfg.eval_hs_threshold)

        point_mu.update(mu, y, valid)
        point_selected.update(selected_pred, y, valid)
        prob_tracker.update(mu, sigma, y, valid)

    result = {
        "mu": point_mu.compute(),
        "selected_pred": point_selected.compute(),
        "prob": prob_tracker.compute(),
    }

    result["mu_improve_mae_vs_selected_percent"] = (
        (result["selected_pred"]["mae"] - result["mu"]["mae"])
        / max(result["selected_pred"]["mae"], 1e-12)
        * 100.0
    )

    result["mu_improve_rmse_vs_selected_percent"] = (
        (result["selected_pred"]["rmse"] - result["mu"]["rmse"])
        / max(result["selected_pred"]["rmse"], 1e-12)
        * 100.0
    )

    return result


def configure_stage1(model: nn.Module):
    model = unwrap_model(model)
    set_requires_grad(model.selector, False)
    set_requires_grad(model.selector.selector, True)
    alpha = getattr(model.selector, "skill_prior_alpha_raw", None)
    if alpha is not None and model.cfg.skill_prior_alpha_learnable:
        alpha.requires_grad_(True)
    set_requires_grad(model.prob_head, False)
    model.selector.train()
    model.selector.wind_encoder.freeze()
    model.prob_head.eval()


def run_selector_epoch(
    model: nn.Module,
    loader: DataLoader,
    cfg: Stage2Config,
    device,
    optimizer: Optional[torch.optim.Optimizer],
    class_weights: Optional[torch.Tensor],
    rank: int,
    epoch: int,
    distributed_reduce: bool,
    desc: str,
) -> Dict[str, Any]:
    is_train = optimizer is not None

    if is_train:
        configure_stage1(model)
    else:
        unwrap_model(model).eval()

    show = is_rank0(rank)
    iterator = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False, disable=not show)

    agg = None
    n_batches = 0

    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        batch = prepare_generated_batch(model, batch)
        static = batch["static"] if "static" in batch else None
        position = batch["position"] if "position" in batch else None

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if isinstance(model, DDP):
                outputs = model(
                    batch["x_in"],
                    batch["m_mod"],
                    static=static,
                    position=position,
                    stage="selector",
                )
            else:
                outputs = model.forward_selector(
                    batch["x_in"], batch["m_mod"], static=static, position=position
                )

            loss, metrics = selector_loss_fn(outputs, batch, cfg, class_weights)

            if is_train:
                loss.backward()
                if cfg.grad_clip is not None and cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
                    )
                optimizer.step()

        if agg is None:
            agg = _init_metric_agg(metrics)
        _update_metric_agg(agg, metrics)
        n_batches += 1

        if show and is_train:
            iterator.set_postfix(
                loss=f"{metrics['loss_total']:.4f}",
                rank=f"{metrics['loss_rank']:.3f}",
                st=f"{metrics['loss_st_hardcost']:.3f}",
                regret=f"{metrics['mean_regret']:.3f}",
                alpha=f"{metrics['skill_prior_alpha']:.2f}",
                refresh=False,
            )

    return _reduce_and_finalize_agg(agg, n_batches, device, distributed_reduce=distributed_reduce)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    cfg: Stage2Config,
    epoch: int,
    best_metric: float,
    path: str,
    stage: str,
    class_weights: Optional[torch.Tensor] = None,
):
    raw_model = unwrap_model(model)
    ckpt = {
        "epoch": epoch,
        "best_metric": best_metric,
        "stage": stage,
        "config": cfg.__dict__,
        "model_state_dict": raw_model.state_dict(),
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if class_weights is not None:
        ckpt["class_weights"] = class_weights.detach().cpu()
    atomic_save(ckpt, path)


def load_checkpoint(model: nn.Module, path: str, device, strict: bool = True):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch_load_safe(path, map_location=device)
    saved_cfg = ckpt.get("config", {})
    if saved_cfg.get("expert_source") != "stage1_generated_v1":
        raise ValueError(
            "This checkpoint was trained with external expert fields; retrain the selector using generated Stage-1 experts."
        )
    if (
        saved_cfg.get("stage1_checkpoint_sha256")
        != unwrap_model(model).cfg.stage1_checkpoint_sha256
    ):
        raise ValueError(
            "Selector checkpoint uses a different/missing frozen Stage-1 encoder; old current-wind selector checkpoints are incompatible."
        )
    state = ckpt["model_state_dict"]
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[len("module.") :]] = v
        else:
            new_state[k] = v
    unwrap_model(model).load_state_dict(new_state, strict=strict)
    return ckpt


def parse_stage2_config():
    cfg = Stage2Config()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--data-h5", default=cfg.data_h5)
    parser.add_argument("--static-npz", default=cfg.static_npz)
    parser.add_argument("--save-dir", default=cfg.save_dir)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size_per_gpu)
    parser.add_argument("--workers", type=int, default=cfg.num_workers)
    parser.add_argument("--epochs", type=int, default=cfg.stage1_epochs)
    parser.add_argument(
        "--encoder-amp",
        action="store_true",
        help="Autocast the frozen encoder AND expert decoders.",
    )
    parser.add_argument("--recompute-class-weights", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.epochs) < 1 or args.workers < 0:
        parser.error("batch-size/epochs must be positive; workers must be nonnegative.")
    cfg.stage1_checkpoint = args.stage1_checkpoint
    cfg.data_h5, cfg.static_npz = args.data_h5, args.static_npz
    cfg.save_dir = args.save_dir
    cfg.batch_size_per_gpu, cfg.num_workers = args.batch_size, args.workers
    cfg.stage1_epochs = args.epochs
    cfg.encoder_amp = args.encoder_amp
    cfg.recompute_class_weights = args.recompute_class_weights
    return cfg


def train_stage2():
    rank, local_rank, world_size, device, distributed = setup_distributed()

    try:
        cfg = parse_stage2_config()
        cfg.device = str(device)

        set_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        if is_rank0(rank):
            ensure_dir(cfg.save_dir)
        barrier()

        if is_rank0(rank):
            cfg, expert_names = update_config_from_data(cfg)
        barrier()
        if not is_rank0(rank):
            cfg, expert_names = update_config_from_data(cfg)
        cfg.device = str(device)

        prepare_stage1_context(cfg, expert_names)

        # Reuse Stage-1 training normalization verbatim
        x_mean, x_std = compute_or_load_wind_stats(cfg, rank)

        train_loader, val_loader, test_loader, train_sampler = build_datasets_and_loaders(
            cfg, x_mean, x_std, rank, world_size, distributed
        )

        backbone = FrozenStage1WindEncoder(cfg).to(device)
        _STAGE1_CONTEXT["backbone"] = backbone
        class_weights_np = compute_or_load_generated_statistics(
            cfg, expert_names, rank, world_size, device, backbone, train_loader.dataset
        )
        class_weights = torch.as_tensor(class_weights_np, dtype=torch.float32, device=device)

        raw_model = SelectorProbabilityModel(cfg).to(device)
        # State tensors have now been copied into the frozen module.
        _STAGE1_CONTEXT["checkpoint"].pop("model", None)
        _STAGE1_CONTEXT.pop("backbone")
        del backbone

        if is_rank0(rank):
            print("\n" + "=" * 90)
            print("Selector-to-Probability Global HDF5 DDP Training")
            print("=" * 90)
            print(f"Device / world_size    : {device} / {world_size}")
            print(f"Data H5                : {cfg.data_h5}")
            print(f"Static NPZ             : {cfg.static_npz}")
            print(f"Static channels        : {cfg.static_channels}")
            print(
                f"Position features      : {cfg.use_position_features} ({cfg.position_channels} ch)"
            )
            print(f"Skill prior            : {cfg.use_expert_skill_prior}")
            if cfg.use_expert_skill_prior:
                print(f"Skill prior NPZ        : {cfg.expert_skill_prior_npz}")
                print(f"Prior as feature       : {cfg.skill_prior_use_as_feature}")
                print(f"Prior logit bias       : {cfg.skill_prior_use_logit_bias}")
                print(f"Prior alpha init       : {cfg.skill_prior_alpha_init}")
                print(f"Prior alpha learnable  : {cfg.skill_prior_alpha_learnable}")
                print(f"Dynamic logits zero-init: {cfg.initialize_dynamic_logits_zero}")
            print(f"Expert anomaly         : {cfg.use_expert_anomaly_features}")
            print(f"Expert norm anomaly    : {cfg.use_expert_normalized_anomaly_features}")
            print(f"in_time                : {cfg.in_time}")
            print(f"num_experts            : {cfg.num_experts}")
            print(f"batch/GPU              : {cfg.batch_size_per_gpu}")
            print(f"global batch            : {cfg.batch_size_per_gpu * world_size}")
            print(f"Total params           : {count_parameters(raw_model, False):,}")
            print(f"Train Hs threshold     : {cfg.train_hs_threshold}")
            print(f"Eval Hs threshold      : {cfg.eval_hs_threshold}")
            print("=" * 90)

        selector_best_path = os.path.join(cfg.save_dir, cfg.selector_best_name)

        # Stage 1
        if cfg.pretrained_selector_ckpt is not None:
            load_checkpoint(raw_model, cfg.pretrained_selector_ckpt, device, strict=True)
            if is_rank0(rank):
                print(f"Loaded pretrained selector: {cfg.pretrained_selector_ckpt}")
        else:
            configure_stage1(raw_model)

            if distributed:
                stage1_model = DDP(
                    raw_model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    broadcast_buffers=True,
                    find_unused_parameters=False,
                )
            else:
                stage1_model = raw_model

            optimizer1 = torch.optim.AdamW(
                [p for p in raw_model.parameters() if p.requires_grad],
                lr=cfg.stage1_lr,
                weight_decay=cfg.weight_decay,
            )

            scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer1,
                mode="min",
                factor=cfg.stage1_scheduler_factor,
                patience=cfg.stage1_scheduler_patience,
                threshold=cfg.stage1_scheduler_threshold,
                threshold_mode="abs",
                min_lr=cfg.stage1_min_lr,
            )

            best_selector_mae = float("inf")
            early_stop_best_mae = float("inf")
            epochs_without_improve = 0
            stage1_history = []

            if is_rank0(rank):
                print("\n" + "=" * 90)
                print("Stage 2: Train selector with frozen Stage-1 encoder + expert decoders")
                print("=" * 90)
                print(f"Selector trainable params: {count_parameters(raw_model, True):,}")
                print(f"lambda_rank            = {cfg.lambda_rank}")
                print(f"lambda_st_hardcost     = {cfg.lambda_st_hardcost}")
                print(f"lambda_soft_oracle     = {cfg.lambda_soft_oracle}")
                print(f"rank_gap_scale_m       = {cfg.rank_gap_scale_m:.3f} m")
                print(f"ranking class weights  = {cfg.ranking_use_class_weights}")
                print(f"rank pair chunk size   = {cfg.rank_pair_chunk_size}")
                print(f"number of expert pairs = {cfg.num_experts * (cfg.num_experts - 1) // 2}")
                print(f"ST temperature         = {cfg.st_temperature:.3f}")
                print(
                    f"soft oracle            = {cfg.use_soft_oracle}, "
                    f"tau={cfg.soft_oracle_temperature:.3f} m"
                )
                print(
                    f"LR scheduler            = ReduceLROnPlateau("
                    f"factor={cfg.stage1_scheduler_factor}, "
                    f"patience={cfg.stage1_scheduler_patience}, "
                    f"min_lr={cfg.stage1_min_lr})"
                )
                print(
                    f"early stopping          = {cfg.stage1_early_stopping}, "
                    f"patience={cfg.stage1_early_stopping_patience}, "
                    f"min_delta={cfg.stage1_early_stopping_min_delta}"
                )
                print(f"selector input channels= {raw_model.selector.selector_in_channels}")
                if cfg.use_expert_skill_prior:
                    print(
                        f"initial prior alpha     = "
                        f"{float(raw_model.selector.get_skill_prior_alpha().detach().cpu()):.4f}"
                    )
                    print(
                        "initialization           = spatial prior + zero dynamic logits"
                        if cfg.initialize_dynamic_logits_zero
                        else "spatial prior + random dynamic logits"
                    )

            for epoch in range(1, cfg.stage1_epochs + 1):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                train_stats = run_selector_epoch(
                    stage1_model,
                    train_loader,
                    cfg,
                    device,
                    optimizer1,
                    class_weights,
                    rank,
                    epoch,
                    distributed_reduce=distributed,
                    desc=f"Stage2 Selector Train {epoch:03d}/{cfg.stage1_epochs:03d}",
                )

                barrier()

                # Validation only on rank0
                val_mae_local = float("inf")
                stop_training_local = False
                record = None
                val_eval = None
                val_stats = None
                improve_mae = float("nan")
                improve_rmse = float("nan")
                val_rmse = float("nan")

                if is_rank0(rank):
                    val_stats = run_selector_epoch(
                        raw_model,
                        val_loader,
                        cfg,
                        device,
                        None,
                        class_weights,
                        rank,
                        epoch,
                        distributed_reduce=False,
                        desc=f"Stage2 Selector Val   {epoch:03d}/{cfg.stage1_epochs:03d}",
                    )
                    val_eval = evaluate_selector(raw_model, val_loader, cfg)

                    val_mae = float(val_eval["selector"]["mae"])
                    val_rmse = float(val_eval["selector"]["rmse"])
                    improve_mae = float(val_eval["selector_improve_mae_vs_best_expert_percent"])
                    improve_rmse = float(val_eval["selector_improve_rmse_vs_best_expert_percent"])
                    val_mae_local = val_mae

                    if val_mae < best_selector_mae:
                        best_selector_mae = val_mae
                        save_checkpoint(
                            raw_model,
                            optimizer1,
                            cfg,
                            epoch,
                            best_selector_mae,
                            selector_best_path,
                            "selector",
                            class_weights,
                        )
                        print(f"  -> Saved selector best to {selector_best_path}")

                    if val_mae < early_stop_best_mae - cfg.stage1_early_stopping_min_delta:
                        early_stop_best_mae = val_mae
                        epochs_without_improve = 0
                    else:
                        epochs_without_improve += 1

                    if (
                        cfg.stage1_early_stopping
                        and epochs_without_improve >= cfg.stage1_early_stopping_patience
                    ):
                        stop_training_local = True

                # DDP-safe: broadcast validation MAE / stop flag
                metric_tensor = torch.tensor(
                    [val_mae_local if is_rank0(rank) else 0.0], dtype=torch.float64, device=device
                )
                stop_tensor = torch.tensor(
                    [1 if (is_rank0(rank) and stop_training_local) else 0],
                    dtype=torch.int32,
                    device=device,
                )

                if distributed:
                    dist.broadcast(metric_tensor, src=0)
                    dist.broadcast(stop_tensor, src=0)

                val_mae_shared = float(metric_tensor.item())
                stop_training = bool(int(stop_tensor.item()))

                scheduler1.step(val_mae_shared)
                current_lr = float(optimizer1.param_groups[0]["lr"])

                if is_rank0(rank):
                    record = {
                        "epoch": epoch,
                        **{f"train_{k}": v for k, v in train_stats.items()},
                        **{f"val_{k}": v for k, v in val_stats.items()},
                        "val_selector_mae": val_mae_local,
                        "val_selector_rmse": val_rmse,
                        "val_improve_mae_percent": improve_mae,
                        "val_improve_rmse_percent": improve_rmse,
                        "val_selector_frequency": val_eval["selector_frequency"],
                        "val_oracle_frequency": val_eval["oracle_frequency"],
                        "lr": current_lr,
                        "epochs_without_improve": int(epochs_without_improve),
                    }
                    stage1_history.append(record)

                    print(
                        f"[Stage1 Epoch {epoch:03d}] "
                        f"lr={current_lr:.3e} "
                        f"train_loss={train_stats['loss_total']:.6f} "
                        f"train_rank={train_stats['loss_rank']:.6f} "
                        f"train_st={train_stats['loss_st_hardcost']:.6f} "
                        f"train_soft={train_stats['loss_soft_oracle']:.6f} "
                        f"train_regret={train_stats['mean_regret']:.6f} "
                        f"train_pairacc={train_stats['pairwise_order_acc']:.4f} "
                        f"prior_alpha={val_stats['skill_prior_alpha']:.4f} "
                        f"val_loss={val_stats['loss_total']:.6f} "
                        f"val_rank={val_stats['loss_rank']:.6f} "
                        f"val_st={val_stats['loss_st_hardcost']:.6f} "
                        f"val_regret={val_stats['mean_regret']:.6f} "
                        f"val_pairacc={val_stats['pairwise_order_acc']:.4f} "
                        f"val_acc={val_stats['acc']:.4f} "
                        f"val_mae={val_mae_local:.6f} "
                        f"val_rmse={val_rmse:.6f} "
                        f"improve_mae={improve_mae:.2f}% "
                        f"improve_rmse={improve_rmse:.2f}% "
                        f"no_improve={epochs_without_improve:02d}/"
                        f"{cfg.stage1_early_stopping_patience:02d} "
                        f"pred_freq={format_array(val_eval['selector_frequency'])} "
                        f"oracle_freq={format_array(val_eval['oracle_frequency'])}"
                    )

                    with open(
                        os.path.join(cfg.save_dir, "stage1_selector_history.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(stage1_history, f, indent=2, ensure_ascii=False)

                    if stop_training:
                        print(
                            f"\nEarly stopping at epoch {epoch}: "
                            f"validation selected MAE has not improved by "
                            f"> {cfg.stage1_early_stopping_min_delta:g} for "
                            f"{cfg.stage1_early_stopping_patience} epochs."
                        )
                        print(f"Best validation selected MAE = {best_selector_mae:.6f}")

                barrier()

                if stop_training:
                    break
                barrier()

            load_checkpoint(raw_model, selector_best_path, device, strict=True)
            barrier()

            if isinstance(stage1_model, DDP):
                del stage1_model

        if is_rank0(rank):
            print("\n" + "=" * 90)
            print("Stage 2: Final selector evaluation")
            print("=" * 90)
            val_selector_final = evaluate_selector(raw_model, val_loader, cfg)
            test_selector_final = evaluate_selector(raw_model, test_loader, cfg)
            stage1_final = {
                "expert_names": expert_names,
                "class_weights": class_weights.detach().cpu().numpy().tolist(),
                "config": cfg.__dict__,
                "parallelism": "DDP_NCCL" if distributed else "single_process",
                "world_size": world_size,
                "val_selector": val_selector_final,
                "test_selector": test_selector_final,
            }
            result_path = os.path.join(cfg.save_dir, "stage2_selector_final_results.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(stage1_final, f, indent=2, ensure_ascii=False)
            print("\nValidation selector:")
            print(json.dumps(val_selector_final, indent=2, ensure_ascii=False))
            print("\nTest selector:")
            print(json.dumps(test_selector_final, indent=2, ensure_ascii=False))
            print(f"\nStage2 selector best checkpoint: {selector_best_path}")
            print(f"Stage2 selector final results : {result_path}")
        barrier()
        return

    finally:
        cleanup_distributed()


def main():
    """CLI entry point for Stage-2 expert selector training."""
    train_stage2()


if __name__ == "__main__":
    main()
