from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from ..models.expert_selector import Stage2Config
from ..models.factory import get_stage1_context
from ..models.probabilistic_forecaster import SelectorProbabilityModel
from .common import (
    atomic_save,
    barrier,
    broadcast_bool,
    broadcast_scalar,
    cleanup_distributed,
    ensure_dir,
    is_rank0,
    move_batch_to_device,
    reduce_weighted_sums,
    set_requires_grad,
    set_seed,
    setup_distributed,
    torch_load_safe,
    unwrap_model,
)
from .train_selector import (
    build_datasets_and_loaders,
    compute_or_load_wind_stats,
    evaluate_probability,
    load_checkpoint,
    make_valid_mask,
    prepare_generated_batch,
    prepare_stage1_context,
    probability_loss_fn,
    update_config_from_data,
)


@dataclass
class Stage3Config:
    # Current dataset / fixed inputs
    data_h5: str = r"./global_wave_dataset_base_1h_0p5deg/global_wave_dataset_base_1h_0p5deg.h5"

    static_npz: str = (
        r"./global_wave_dataset_base_1h_0p5deg/"
        r"static_etopo5_3features_global_h5grid_0p5deg.npz"
    )

    # Locations may change; pretrained contents must match the Stage-2 source.
    stage1_checkpoint: str = "./outputs_wave_experts_stage1_new/best.pt"
    selector_checkpoint: str = "./outputs_stage2_generated_experts/stage2_selector_best.pt"

    # Stage 3 training
    epochs: int = 200
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-6
    grad_clip: float = 5.0

    # Per GPU on the 0.5-degree grid; reduce 4 -> 2 -> 1 if needed.
    batch_size_per_gpu: int = 4
    num_workers: int = 0

    lambda_nll: float = 1.0
    lambda_mu_mae: float = 0.3
    min_sigma: float = 0.05

    # LR schedule / early stopping use validation NLL because this is the
    # probability stage. MAE-best is saved independently.
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    scheduler_threshold: float = 1.0e-5
    min_lr: float = 1.0e-6

    early_stopping: bool = True
    early_stopping_patience: int = 35
    early_stopping_min_delta: float = 1.0e-5

    # Reproducibility / validation
    seed: int = 42
    strict_selector_checkpoint: bool = True

    expected_num_experts: Optional[int] = 21
    expected_h: Optional[int] = 281
    expected_w: Optional[int] = 720
    expected_time_step_hours: Optional[int] = 1

    # Output
    save_dir: str = "./outputs_stage3_probability"

    best_nll_name: str = "stage3_probability_best_nll.pt"
    best_mae_name: str = "stage3_probability_best_mae.pt"
    final_name: str = "final_selector_probability_best_nll.pt"

    history_name: str = "stage3_probability_history.json"
    final_results_name: str = "stage3_final_results.json"


def save_full_checkpoint(
    model: SelectorProbabilityModel,
    optimizer: torch.optim.Optimizer,
    cfg_model: Stage2Config,
    cfg_stage3: Stage3Config,
    epoch: int,
    best_metric: float,
    metric_name: str,
    path: str,
) -> None:
    ckpt = {
        "epoch": int(epoch),
        "stage": "probability",
        "metric_name": str(metric_name),
        "best_metric": float(best_metric),
        "stage1_checkpoint": os.path.abspath(cfg_stage3.stage1_checkpoint),
        "config": dict(cfg_model.__dict__),
        "stage3_config": dict(cfg_stage3.__dict__),
        "selector_checkpoint": os.path.abspath(cfg_stage3.selector_checkpoint),
        "expert_source": "stage1_generated_v1",
        "expert_names": list(get_stage1_context()["checkpoint"]["expert_names"]),
        "stage1_architecture": dict(cfg_model.stage1_architecture),
        "input_normalization": dict(
            mean=get_stage1_context()["mean"], std=get_stage1_context()["std"]
        ),
        "ocean_mask": get_stage1_context()["ocean"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    atomic_save(ckpt, path)


def build_model_config_from_checkpoints(run_cfg, device):
    """Restore model architecture and exact generated-expert prior from Stage 2."""
    if not os.path.isfile(run_cfg.selector_checkpoint):
        raise FileNotFoundError(run_cfg.selector_checkpoint)
    ckpt = torch_load_safe(run_cfg.selector_checkpoint, map_location="cpu")
    saved_cfg = ckpt.get("config", {})
    if saved_cfg.get("expert_source") != "stage1_generated_v1":
        raise ValueError(
            "Expected the corrected Stage-2 checkpoint trained on generated Stage-1 experts."
        )
    if ckpt.get("stage") == "probability":
        raise ValueError("Use the Stage-2 SELECTOR checkpoint as --selector-checkpoint.")
    cfg = Stage2Config()
    for key, value in saved_cfg.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    original_arch = dict(cfg.stage1_architecture)
    original_sha = cfg.stage1_checkpoint_sha256
    original_history = cfg.in_time
    original_latent = cfg.wind_base_channels
    cfg.stage1_checkpoint = run_cfg.stage1_checkpoint
    cfg.data_h5, cfg.static_npz = run_cfg.data_h5, run_cfg.static_npz
    cfg.device = str(device)
    cfg.batch_size_per_gpu, cfg.num_workers = run_cfg.batch_size_per_gpu, run_cfg.num_workers
    cfg.save_dir = run_cfg.save_dir
    cfg.weight_decay, cfg.grad_clip = run_cfg.weight_decay, run_cfg.grad_clip
    cfg.lambda_nll, cfg.lambda_mu_mae, cfg.min_sigma = (
        run_cfg.lambda_nll,
        run_cfg.lambda_mu_mae,
        run_cfg.min_sigma,
    )
    cfg.expected_num_experts = run_cfg.expected_num_experts
    cfg.expected_h, cfg.expected_w = run_cfg.expected_h, run_cfg.expected_w
    cfg.expected_time_step_hours = run_cfg.expected_time_step_hours
    cfg, names = update_config_from_data(cfg)
    prepare_stage1_context(cfg, names)
    if (
        cfg.stage1_checkpoint_sha256 != original_sha
        or cfg.stage1_architecture != original_arch
        or cfg.in_time != original_history
        or cfg.wind_base_channels != original_latent
    ):
        raise ValueError("Stage-1 checkpoint does not match the one used for Stage-2 training.")
    state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in ckpt["model_state_dict"].items()
    }
    saved_geometry = state.get("selector.wind_encoder.geometry")
    geometry = torch.from_numpy(get_stage1_context()["geometry"][None])
    if saved_geometry is None or not torch.equal(saved_geometry.cpu(), geometry):
        raise ValueError("Static/position/ocean geometry differs from the frozen Stage-2 inputs.")
    # Every first-stage parameter, including all decoders, must be identical.
    for key, value in get_stage1_context()["checkpoint"]["model"].items():
        plain = key[7:] if key.startswith("module.") else key
        expected = state.get("selector.wind_encoder.core." + plain)
        if expected is None or not torch.equal(expected.cpu(), value.cpu()):
            raise ValueError("Frozen Stage-1 weight mismatch in selector checkpoint: " + plain)
    return cfg, dict(checkpoint=ckpt, expert_names=names)


def stage3_forward(
    raw_model: SelectorProbabilityModel, train_head: nn.Module, batch: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    static = batch.get("static", None)
    position = batch.get("position", None)

    # Both prior stages are frozen; generate their inputs without autograd.
    with torch.no_grad():
        generated = prepare_generated_batch(raw_model, batch)
        selector_outputs = raw_model.forward_selector(
            generated["x_in"], generated["m_mod"], static=static, position=position
        )
        prob_in = raw_model.build_prob_input(selector_outputs, static)

    prob_outputs = train_head(prob_in, selector_outputs["selected_pred"])

    return {**selector_outputs, **prob_outputs}


def train_one_epoch(
    raw_model: SelectorProbabilityModel,
    train_head: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    cfg_model: Stage2Config,
    run_cfg: Stage3Config,
    device,
    rank: int,
    distributed: bool,
    epoch: int,
) -> Dict[str, float]:
    raw_model.selector.eval()
    train_head.train()

    # Accumulate exact valid-pixel weighted metrics rather than simple
    # unweighted average of batches.
    sums = {"loss_total": 0.0, "loss_nll": 0.0, "loss_mu_mae": 0.0, "sigma_mean": 0.0}
    total_valid = 0.0

    iterator = tqdm(
        loader,
        desc=f"Stage3 Train {epoch:03d}/{run_cfg.epochs:03d}",
        dynamic_ncols=True,
        leave=False,
        disable=not is_rank0(rank),
    )

    for batch in iterator:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        outputs = stage3_forward(raw_model=raw_model, train_head=train_head, batch=batch)

        loss, metrics = probability_loss_fn(outputs, batch, cfg_model)

        with torch.no_grad():
            local_count = (
                make_valid_mask(
                    outputs["mu"], batch["y_sat"], batch["mask"], cfg_model.train_hs_threshold
                )
                .sum()
                .double()
            )
            global_count = local_count.clone()
            if distributed:
                dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
            ranks = dist.get_world_size() if distributed else 1
        loss = loss * (local_count / global_count.clamp_min(1)).to(loss.dtype) * ranks
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Stage-3 training loss.")
        loss.backward()
        if run_cfg.grad_clip is not None and run_cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                [p for p in unwrap_model(train_head).parameters() if p.requires_grad],
                max_norm=float(run_cfg.grad_clip),
            )
        if global_count.item() > 0:
            optimizer.step()

        with torch.no_grad():
            valid = make_valid_mask(
                outputs["mu"], batch["y_sat"], batch["mask"], cfg_model.train_hs_threshold
            )
            n_valid = float(valid.sum().item())

        if n_valid > 0:
            total_valid += n_valid
            sums["loss_total"] += float(metrics["loss_total"]) * n_valid
            sums["loss_nll"] += float(metrics["loss_nll"]) * n_valid
            sums["loss_mu_mae"] += float(metrics["loss_mu_mae"]) * n_valid
            sums["sigma_mean"] += float(metrics["sigma_mean"]) * n_valid

        if is_rank0(rank):
            iterator.set_postfix(
                loss=f"{metrics['loss_total']:.4f}",
                nll=f"{metrics['loss_nll']:.4f}",
                mae=f"{metrics['loss_mu_mae']:.4f}",
                sigma=f"{metrics['sigma_mean']:.3f}",
                refresh=False,
            )

    return reduce_weighted_sums(sums, total_valid, device=device, distributed=distributed)


def parse_stage3_config():
    cfg = Stage3Config()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--stage1-checkpoint", required=True)
    p.add_argument("--selector-checkpoint", required=True)
    p.add_argument("--data-h5", default=cfg.data_h5)
    p.add_argument("--static-npz", default=cfg.static_npz)
    p.add_argument("--save-dir", default=cfg.save_dir)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size_per_gpu)
    p.add_argument("--workers", type=int, default=cfg.num_workers)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.lr)
    args = p.parse_args()
    if min(args.batch_size, args.epochs) < 1 or args.workers < 0 or args.lr <= 0:
        p.error("batch-size/epochs/lr must be positive; workers must be nonnegative.")
    cfg.stage1_checkpoint, cfg.selector_checkpoint = (
        args.stage1_checkpoint,
        args.selector_checkpoint,
    )
    cfg.data_h5, cfg.static_npz = args.data_h5, args.static_npz
    cfg.save_dir = args.save_dir
    cfg.batch_size_per_gpu, cfg.num_workers = args.batch_size, args.workers
    cfg.epochs, cfg.lr = args.epochs, args.lr
    return cfg


def train_stage3() -> None:
    rank, local_rank, world_size, device, distributed = setup_distributed()

    try:
        run_cfg = parse_stage3_config()
        set_seed(run_cfg.seed + rank)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        if is_rank0(rank):
            ensure_dir(run_cfg.save_dir)
        barrier()

        # Restore the Stage-1 architecture and exact Stage-2 selector state.
        cfg_model, meta = build_model_config_from_checkpoints(run_cfg, device)
        expert_names = meta["expert_names"]

        # The Stage-1 checkpoint supplies authoritative full-history wind stats.
        x_mean, x_std = compute_or_load_wind_stats(cfg_model, rank)

        train_loader, val_loader, test_loader, train_sampler = build_datasets_and_loaders(
            cfg_model, x_mean, x_std, rank, world_size, distributed
        )

        # Load Stage 2 strictly, including its frozen full Stage-1 model.
        prior = None
        if cfg_model.use_expert_skill_prior:
            prior = tuple(
                meta["checkpoint"]["model_state_dict"][key][0].cpu().numpy()
                for key in (
                    "selector.skill_prior_logits_fixed",
                    "selector.skill_prior_confidence_fixed",
                )
            )
        raw_model = SelectorProbabilityModel(cfg_model, prior=prior).to(device)
        load_checkpoint(
            raw_model,
            run_cfg.selector_checkpoint,
            device,
            strict=run_cfg.strict_selector_checkpoint,
        )

        get_stage1_context()["checkpoint"].pop("model", None)
        del meta["checkpoint"]
        set_requires_grad(raw_model.selector, False)
        raw_model.selector.eval()

        # Keep the probability-head initialization stored with Stage 2.
        set_requires_grad(raw_model.prob_head, True)
        raw_model.prob_head.train()

        # Only wrap/train the probability head with DDP. The huge frozen
        # selector is not part of DDP gradient synchronization.
        if distributed:
            train_head: nn.Module = DDP(
                raw_model.prob_head,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=True,
                find_unused_parameters=False,
            )
        else:
            train_head = raw_model.prob_head

        optimizer = torch.optim.AdamW(
            raw_model.prob_head.parameters(), lr=run_cfg.lr, weight_decay=run_cfg.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=run_cfg.scheduler_factor,
            patience=run_cfg.scheduler_patience,
            threshold=run_cfg.scheduler_threshold,
            threshold_mode="abs",
            min_lr=run_cfg.min_lr,
        )

        best_nll = float("inf")
        best_mae = float("inf")
        early_best_nll = float("inf")
        epochs_without_improve = 0
        history = []

        best_nll_path = os.path.join(run_cfg.save_dir, run_cfg.best_nll_name)
        best_mae_path = os.path.join(run_cfg.save_dir, run_cfg.best_mae_name)
        final_path = os.path.join(run_cfg.save_dir, run_cfg.final_name)
        history_path = os.path.join(run_cfg.save_dir, run_cfg.history_name)
        final_results_path = os.path.join(run_cfg.save_dir, run_cfg.final_results_name)

        if is_rank0(rank):
            selector_params = sum(p.numel() for p in raw_model.selector.parameters())
            head_params = sum(p.numel() for p in raw_model.prob_head.parameters())
            trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)

            print("\n" + "=" * 94)
            print("Stage 3: Frozen Stage-1 experts + Stage-2 selector -> Probability Head")
            print("=" * 94)
            print(f"Device/world size      : {device} / {world_size}")
            print(f"Data H5                : {cfg_model.data_h5}")
            print(f"Static                 : {cfg_model.static_npz}")
            print(f"Skill prior            : {cfg_model.expert_skill_prior_npz}")
            print(f"Stage-1 checkpoint     : {run_cfg.stage1_checkpoint}")
            print(f"Stage-2 selector       : {run_cfg.selector_checkpoint}")
            print(f"Grid                    : {run_cfg.expected_h} x {run_cfg.expected_w}")
            print(f"Experts                 : {len(expert_names)}")
            print(f"Expert names            : {expert_names}")
            print(f"Batch/GPU               : {run_cfg.batch_size_per_gpu}")
            print(f"Global batch            : {run_cfg.batch_size_per_gpu * world_size}")
            print(f"Frozen selector params  : {selector_params:,}")
            print(f"Probability-head params : {head_params:,}")
            print(f"Trainable params        : {trainable:,}")
            print(f"LR                      : {run_cfg.lr:.3e}")
            print(f"lambda_nll              : {run_cfg.lambda_nll}")
            print(f"lambda_mu_mae           : {run_cfg.lambda_mu_mae}")
            print(f"min_sigma               : {run_cfg.min_sigma}")
            print(
                "Stage-3 input            : z + selected_pred + confidence + std + range + static3 + onehot"
            )
            print("Best checkpoints         : validation NLL AND validation mu MAE")
            print("=" * 94)

        # Train
        for epoch in range(1, run_cfg.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(20000 + epoch)

            train_stats = train_one_epoch(
                raw_model=raw_model,
                train_head=train_head,
                loader=train_loader,
                optimizer=optimizer,
                cfg_model=cfg_model,
                run_cfg=run_cfg,
                device=device,
                rank=rank,
                distributed=distributed,
                epoch=epoch,
            )

            barrier()

            val_nll_local = 0.0
            stop_local = False

            if is_rank0(rank):
                # raw_model.prob_head is the underlying DDP module and already
                # contains the synchronized trained weights.
                raw_model.selector.eval()
                raw_model.prob_head.eval()

                val_eval = evaluate_probability(raw_model, val_loader, cfg_model)

                val_nll = float(val_eval["prob"]["nll"])
                val_crps = float(val_eval["prob"]["crps"])
                val_mu_mae = float(val_eval["mu"]["mae"])
                val_mu_rmse = float(val_eval["mu"]["rmse"])
                val_selected_mae = float(val_eval["selected_pred"]["mae"])
                val_selected_rmse = float(val_eval["selected_pred"]["rmse"])
                improve_mae = float(val_eval["mu_improve_mae_vs_selected_percent"])
                improve_rmse = float(val_eval["mu_improve_rmse_vs_selected_percent"])
                val_nll_local = val_nll

                saved_nll = False
                saved_mae = False

                if val_nll < best_nll:
                    best_nll = val_nll
                    save_full_checkpoint(
                        raw_model,
                        optimizer,
                        cfg_model,
                        run_cfg,
                        epoch,
                        best_nll,
                        "val_nll",
                        best_nll_path,
                    )
                    temporary_final = final_path + ".tmp"
                    shutil.copyfile(best_nll_path, temporary_final)
                    os.replace(temporary_final, final_path)
                    saved_nll = True

                if val_mu_mae < best_mae:
                    best_mae = val_mu_mae
                    save_full_checkpoint(
                        raw_model,
                        optimizer,
                        cfg_model,
                        run_cfg,
                        epoch,
                        best_mae,
                        "val_mu_mae",
                        best_mae_path,
                    )
                    saved_mae = True

                # Early stop on probability objective (NLL).
                if val_nll < early_best_nll - run_cfg.early_stopping_min_delta:
                    early_best_nll = val_nll
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1

                if (
                    run_cfg.early_stopping
                    and epochs_without_improve >= run_cfg.early_stopping_patience
                ):
                    stop_local = True

                current_lr = float(optimizer.param_groups[0]["lr"])
                record = {
                    "epoch": epoch,
                    "lr": current_lr,
                    **{f"train_{k}": float(v) for k, v in train_stats.items()},
                    "val_nll": val_nll,
                    "val_crps": val_crps,
                    "val_mu_mae": val_mu_mae,
                    "val_mu_rmse": val_mu_rmse,
                    "val_selected_mae": val_selected_mae,
                    "val_selected_rmse": val_selected_rmse,
                    "val_mu_improve_mae_vs_selected_percent": improve_mae,
                    "val_mu_improve_rmse_vs_selected_percent": improve_rmse,
                    "val_picp90": float(val_eval["prob"]["picp90"]),
                    "val_mpiw90": float(val_eval["prob"]["mpiw90"]),
                    "val_picp95": float(val_eval["prob"]["picp95"]),
                    "val_mpiw95": float(val_eval["prob"]["mpiw95"]),
                    "val_sigma_mean": float(val_eval["prob"]["sigma_mean"]),
                    "best_val_nll": float(best_nll),
                    "best_val_mu_mae": float(best_mae),
                    "epochs_without_nll_improve": int(epochs_without_improve),
                }
                history.append(record)

                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)

                tags = []
                if saved_nll:
                    tags.append("BEST_NLL")
                if saved_mae:
                    tags.append("BEST_MAE")
                tag_text = (" [" + ",".join(tags) + "]") if tags else ""

                print(
                    f"[Stage3 Epoch {epoch:03d}] "
                    f"lr={current_lr:.3e} "
                    f"train_loss={train_stats['loss_total']:.6f} "
                    f"train_nll={train_stats['loss_nll']:.6f} "
                    f"train_mu_mae={train_stats['loss_mu_mae']:.6f} "
                    f"val_nll={val_nll:.6f} "
                    f"val_crps={val_crps:.6f} "
                    f"mu_mae={val_mu_mae:.6f} "
                    f"selected_mae={val_selected_mae:.6f} "
                    f"mu_improve={improve_mae:.2f}% "
                    f"picp90={val_eval['prob']['picp90']:.3f} "
                    f"sigma={val_eval['prob']['sigma_mean']:.4f} "
                    f"no_nll_improve={epochs_without_improve:02d}/"
                    f"{run_cfg.early_stopping_patience:02d}"
                    f"{tag_text}"
                )

            # All ranks must use the same scheduler metric and stop flag.
            val_nll_shared = broadcast_scalar(
                val_nll_local if is_rank0(rank) else 0.0, device, src=0
            )
            stop_training = broadcast_bool(stop_local if is_rank0(rank) else False, device, src=0)

            scheduler.step(val_nll_shared)

            barrier()

            if stop_training:
                if is_rank0(rank):
                    print(
                        f"\nEarly stopping at epoch {epoch}. "
                        f"Best validation NLL={best_nll:.6f}, "
                        f"best validation mu MAE={best_mae:.6f}."
                    )
                break

        barrier()

        # Final evaluation: evaluate BOTH best-NLL and best-MAE models.
        # Rank 0 only; other ranks wait.
        if is_rank0(rank):
            final_results: Dict[str, Any] = {
                "expert_names": expert_names,
                "stage1_checkpoint": run_cfg.stage1_checkpoint,
                "selector_checkpoint": run_cfg.selector_checkpoint,
                "best_nll_checkpoint": best_nll_path,
                "best_mae_checkpoint": best_mae_path,
                "default_final_checkpoint": final_path,
                "best_validation_nll": float(best_nll),
                "best_validation_mu_mae": float(best_mae),
                "models": {},
            }

            for label, path in [("best_nll", best_nll_path), ("best_mae", best_mae_path)]:
                # Stage-3 checkpoints retain the same complete model-state layout.
                load_checkpoint(raw_model, path, device, strict=True)
                raw_model.eval()

                val_result = evaluate_probability(raw_model, val_loader, cfg_model)
                test_result = evaluate_probability(raw_model, test_loader, cfg_model)

                final_results["models"][label] = {"val": val_result, "test": test_result}

                print("\n" + "-" * 94)
                print(f"Final evaluation: {label}")
                print("-" * 94)
                print(
                    f"VAL : selected MAE={val_result['selected_pred']['mae']:.6f}, "
                    f"mu MAE={val_result['mu']['mae']:.6f}, "
                    f"NLL={val_result['prob']['nll']:.6f}, "
                    f"CRPS={val_result['prob']['crps']:.6f}, "
                    f"PICP90={val_result['prob']['picp90']:.3f}"
                )
                print(
                    f"TEST: selected MAE={test_result['selected_pred']['mae']:.6f}, "
                    f"mu MAE={test_result['mu']['mae']:.6f}, "
                    f"NLL={test_result['prob']['nll']:.6f}, "
                    f"CRPS={test_result['prob']['crps']:.6f}, "
                    f"PICP90={test_result['prob']['picp90']:.3f}"
                )
                print(
                    f"TEST deterministic improvement over Stage-2 selection: "
                    f"{test_result['mu_improve_mae_vs_selected_percent']:.2f}%"
                )

            with open(final_results_path, "w", encoding="utf-8") as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)

            print("\n" + "=" * 94)
            print("Stage 3 completed")
            print("=" * 94)
            print(f"Best NLL model : {best_nll_path}")
            print(f"Best MAE model : {best_mae_path}")
            print(f"Default final  : {final_path}")
            print(f"History        : {history_path}")
            print(f"Final results  : {final_results_path}")

        barrier()

    finally:
        cleanup_distributed()


def main():
    """CLI entry point for Stage-3 probabilistic head training."""
    train_stage3()


if __name__ == "__main__":
    main()
