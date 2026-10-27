from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm as stage1_tqdm

from ..data.hdf5 import DayBlockSampler, HistoryDataset, cached_wind_stats, inspect_data, loader
from ..models.expert_forecaster import WaveExpertModel, autocast_context
from .common import (
    capture_rng,
    collective_save,
    dataset_signature,
    gather_rank_state,
    load_pt,
    make_grad_scaler,
    reduce_sum,
    resolve_resume,
    restore_rng,
    set_seed,
    setup,
    unwrap_model,
)


def run_epoch(
    model,
    batches,
    device,
    world,
    optimizer=None,
    scaler=None,
    amp=False,
    area_weighted=False,
    lambda_anomaly=0.0,
    resume_state=None,
    on_batch_end=None,
    start_batch=0,
    total_batches=None,
):
    training = optimizer is not None
    model.train(training)
    k = len(unwrap_model(model).decoders)
    aggregate = torch.zeros(6, k, dtype=torch.float64, device=device)
    diversity = torch.zeros(5, dtype=torch.float64, device=device)
    extreme = torch.zeros(3, 3, k, dtype=torch.float64, device=device)
    if resume_state is not None:
        for name, tensor in [
            ("aggregate", aggregate),
            ("diversity", diversity),
            ("extreme", extreme),
        ]:
            saved = resume_state[name]
            if tuple(saved.shape) != tuple(tensor.shape):
                raise ValueError(f"Resume metric shape differs: {name}")
            tensor.copy_(saved.to(device=device, dtype=tensor.dtype))
    progress = stage1_tqdm(
        batches,
        total=total_batches,
        initial=start_batch,
        desc="Train" if training else "Evaluate",
        disable=dist.is_initialized() and dist.get_rank() != 0,
    )
    for batch_number, batch in enumerate(progress, start=start_batch + 1):
        wind, static, target, valid = [
            batch[key].to(device, non_blocking=True)
            for key in ["wind", "static", "target", "valid"]
        ]
        with torch.set_grad_enabled(training):
            with autocast_context(device.type, amp):
                pred = model(wind, static)
            error = pred.float() - target
            sums = (error.abs() * valid).sum((0, 2, 3))
            counts = valid.sum((0, 2, 3)).double()
            area = batch["area"].to(device)[:, None]
            weights = valid * (area if area_weighted else 1.0)
            weighted_counts = weights.sum((0, 2, 3)).double()
            weighted_sums = (error.abs() * weights).sum((0, 2, 3))
            if training:
                global_counts = reduce_sum(weighted_counts.clone())
                if torch.any(global_counts == 0):
                    raise ValueError("Expert has no valid target in a global training batch.")
                # DDP averages gradients; world corrects this to global per-expert MAE.
                loss = (weighted_sums / global_counts.to(sums.dtype)).mean() * world
                if lambda_anomaly:
                    denom = valid.sum(1, keepdim=True).clamp_min(1)
                    pa = pred.float() - (pred.float() * valid).sum(1, keepdim=True) / denom
                    ta = target - (target * valid).sum(1, keepdim=True) / denom
                    anomaly_sums = ((pa - ta).abs() * weights).sum((0, 2, 3))
                    loss = (
                        loss
                        + lambda_anomaly
                        * (anomaly_sums / global_counts.to(sums.dtype)).mean()
                        * world
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss.")
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        aggregate[0] += sums.detach().double()
        aggregate[1] += ((error.detach().double() ** 2) * valid).sum((0, 2, 3))
        aggregate[2] += counts
        av = valid * area
        aggregate[3] += (error.detach().abs() * av).sum((0, 2, 3)).double()
        aggregate[4] += (error.detach().double() ** 2 * av).sum((0, 2, 3))
        aggregate[5] += av.sum((0, 2, 3)).double()
        common = valid.all(1)
        ps = pred.detach().float().std(1, unbiased=False)
        ts = target.std(1, unbiased=False)
        pa = pred.detach().float() - pred.detach().float().mean(1, keepdim=True)
        ta = target - target.mean(1, keepdim=True)
        diversity[0] += ps[common].double().sum()
        diversity[1] += ts[common].double().sum()
        diversity[2] += (ps - ts).abs()[common].double().sum()
        diversity[3] += (pa - ta).abs().mean(1)[common].double().sum()
        diversity[4] += common.sum()
        for j, threshold in enumerate([4.0, 6.0, 8.0]):
            severe = valid & (target >= threshold)
            extreme[j, 0] += (error.detach().abs() * severe).sum((0, 2, 3)).double()
            extreme[j, 1] += (error.detach().double() ** 2 * severe).sum((0, 2, 3))
            extreme[j, 2] += severe.sum((0, 2, 3)).double()
        if training and on_batch_end is not None:
            # Callback runs on ALL ranks at the same completed optimizer boundary.
            on_batch_end(batch_number, aggregate, diversity, extreme)
    reduce_sum(aggregate)
    if torch.any(aggregate[2] == 0):
        raise ValueError("Expert has no valid samples in this split.")
    mae = aggregate[0] / aggregate[2]
    rmse = torch.sqrt(aggregate[1] / aggregate[2])
    area_mae = aggregate[3] / aggregate[5].clamp_min(1e-12)
    area_rmse = torch.sqrt(aggregate[4] / aggregate[5].clamp_min(1e-12))
    reduce_sum(diversity)
    reduce_sum(extreme)
    severe_metrics = {}
    for j, threshold in enumerate([4, 6, 8]):
        es, ss, count = extreme[j].cpu().numpy()
        severe_metrics[f"WW3_Hs_ge_{threshold}m"] = dict(
            count=count.tolist(),
            mae=[float(es[i] / count[i]) if count[i] else None for i in range(k)],
            rmse=[float(np.sqrt(ss[i] / count[i])) if count[i] else None for i in range(k)],
        )
    d = diversity.cpu().numpy()
    diagnostics = dict(
        common_count=int(d[4]),
        predicted_spread_mean=float(d[0] / d[4]) if d[4] else None,
        ww3_spread_mean=float(d[1] / d[4]) if d[4] else None,
        spread_mae=float(d[2] / d[4]) if d[4] else None,
        expert_anomaly_mae=float(d[3] / d[4]) if d[4] else None,
        spread_ratio=float(d[0] / d[1]) if d[1] > 0 else None,
    )
    return dict(
        mean_mae=(area_mae if area_weighted else mae).mean().item(),
        unweighted_mean_mae=mae.mean().item(),
        area_mean_mae=area_mae.mean().item(),
        mae=mae.cpu().tolist(),
        rmse=rmse.cpu().tolist(),
        area_mae=area_mae.cpu().tolist(),
        area_rmse=area_rmse.cpu().tolist(),
        count=aggregate[2].cpu().tolist(),
        diversity=diagnostics,
        extremes=severe_metrics,
    )


def train_stage1(args):
    rank, world, device = setup()
    try:
        amp = args.amp and device.type == "cuda"
        scaler = make_grad_scaler(amp)
        if rank == 0:
            print(f"PyTorch {torch.__version__} | device={device} | AMP={amp}")
        set_seed(args.seed)
        meta, splits, _, static, ocean, names = inspect_data(args)
        output = Path(args.output)
        checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output
        output.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        signature = dataset_signature(args.data_h5, static, ocean, names)
        resume_path = resolve_resume(args.resume, checkpoint_dir, output)
        checkpoint = load_pt(resume_path) if resume_path is not None else None
        if checkpoint is not None:
            if checkpoint["architecture"] != meta or checkpoint["expert_names"] != names:
                raise ValueError("Resume architecture/expert order differs.")
            moved = checkpoint["data_h5"] != str(Path(args.data_h5).resolve())
            if moved and not args.allow_data_relocation:
                raise ValueError(
                    "Resume source path differs. For an IDENTICAL copied dataset, "
                    "pass --allow-data-relocation and --data-h5 NEW_PATH."
                )
            if "data_signature" in checkpoint and checkpoint["data_signature"] != signature:
                raise ValueError("Resume grid/time/splits/static signature differs.")
            if not np.array_equal(ocean, checkpoint["ocean_mask"]):
                raise ValueError("Resume ocean mask differs.")
            if moved and "data_signature" not in checkpoint and rank == 0:
                print(
                    "Legacy checkpoint has no grid/time/split signature; relocation relies on your identical-copy declaration."
                )
            old_args = checkpoint["args"]
            if (
                old_args["area_weighted"] != args.area_weighted
                or old_args["lambda_anomaly"] != args.lambda_anomaly
            ):
                raise ValueError("Resume loss configuration differs.")
            mean, std = np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"])
        else:
            if rank == 0:
                mean, std = cached_wind_stats(
                    args.data_h5,
                    splits["train"],
                    ocean,
                    output / "wind_stats_train_only_v2.npz",
                    meta["wind_channels"],
                    args.recompute_wind_stats,
                )
            else:
                mean = std = np.empty(meta["wind_channels"], dtype=np.float32)
            if world > 1:
                stats = torch.tensor(np.stack([mean, std]), device=device)
                dist.broadcast(stats, src=0)
                mean, std = stats.cpu().numpy()
        model = WaveExpertModel(**meta).to(device)
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model"], strict=True)
        if world > 1:
            model = DDP(model, device_ids=[device.index], broadcast_buffers=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=10, min_lr=1e-6
        )
        best, stale, start, history = math.inf, 0, 1, []
        start_batch, resume_metrics, resume_rng = 0, None, None
        dataset = HistoryDataset(
            args.data_h5, splits["train"], args.history_hours, static, ocean, mean, std
        )
        sampler = DayBlockSampler(dataset, world, rank, args.batch_size, args.seed)
        total_batches = sampler.total // (world * args.batch_size)
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if amp == checkpoint.get("amp_enabled", checkpoint["args"].get("amp", False)):
                scaler.load_state_dict(checkpoint["scaler"])
            best, stale, history = checkpoint["best"], checkpoint["stale"], checkpoint["history"]
            partial = checkpoint.get("phase", "epoch_complete") == "train"
            start = int(checkpoint["epoch"]) if partial else int(checkpoint["epoch"]) + 1
            states = checkpoint.get("rank_states")
            if partial:
                if states is None or checkpoint["world_size"] != world:
                    raise ValueError(
                        "Mid-epoch resume requires the original number of GPUs/processes."
                    )
                if checkpoint.get("amp_enabled", checkpoint["args"].get("amp", False)) != amp:
                    raise ValueError(
                        "Mid-epoch resume requires unchanged effective AMP mode/device support."
                    )
                for key in ["batch_size", "seed", "amp"]:
                    if checkpoint["args"][key] != vars(args)[key]:
                        raise ValueError(
                            f"Mid-epoch resume requires unchanged --{key.replace('_', '-')}."
                        )
                if checkpoint["total_batches"] != total_batches:
                    raise ValueError("Mid-epoch batch schedule differs.")
                start_batch = int(checkpoint["next_batch"])
                sampler.set_start_batch(start_batch)
                resume_metrics = states[rank]["metrics"]
            if states is not None and checkpoint.get("world_size") == world:
                resume_rng = states[rank]["rng"]
            if rank == 0:
                print(
                    f"Resuming {resume_path}: epoch={start}, completed batches={start_batch}/{total_batches}",
                    flush=True,
                )
        if rank == 0:
            print(
                f"Training drops {len(dataset) - sampler.total} targets per epoch to avoid padded duplicates."
            )
            print(
                f"Checkpoint directory: {checkpoint_dir.resolve()}; periodic interval={args.save_every_batches} batches"
            )
        train_loader = loader(dataset, args, sampler)

        def evaluate(split):
            ds = HistoryDataset(
                args.data_h5,
                splits[split][rank::world],
                args.history_hours,
                static,
                ocean,
                mean,
                std,
            )
            return run_epoch(
                unwrap_model(model),
                loader(ds, args),
                device,
                world,
                amp=amp,
                area_weighted=args.area_weighted,
            )

        def make_state(epoch, phase, next_batch, rank_states):
            return dict(
                format_version=3,
                model=unwrap_model(model).state_dict(),
                architecture=meta,
                expert_names=names,
                mean=mean,
                std=std,
                ocean_mask=ocean,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                epoch=epoch,
                phase=phase,
                next_batch=next_batch,
                best=best,
                stale=stale,
                history=history,
                args=vars(args),
                data_h5=str(Path(args.data_h5).resolve()),
                data_signature=signature,
                world_size=world,
                total_batches=total_batches,
                rank_states=rank_states,
                amp_enabled=amp,
            )

        for epoch in range(start, args.epochs + 1):
            sampler.set_epoch(epoch)
            sampler.set_start_batch(start_batch)
            if resume_rng is not None:
                restore_rng(resume_rng, device)
                resume_rng = None

            def save_batch(completed, aggregate, diversity, extreme):
                if completed % args.save_every_batches and completed != total_batches:
                    return
                local = dict(
                    rng=capture_rng(device),
                    metrics=dict(
                        aggregate=aggregate.detach().cpu(),
                        diversity=diversity.detach().cpu(),
                        extreme=extreme.detach().cpu(),
                    ),
                )
                rank_states = gather_rank_state(local, world)
                collective_save(
                    lambda: make_state(epoch, "train", completed, rank_states),
                    [checkpoint_dir / "last_batch.pt"],
                    rank,
                    world,
                    device,
                )
                if rank == 0:
                    print(
                        f"\nSaved batch checkpoint: epoch {epoch}, {completed}/{total_batches}",
                        flush=True,
                    )

            training = run_epoch(
                model,
                train_loader,
                device,
                world,
                optimizer,
                scaler,
                amp,
                args.area_weighted,
                args.lambda_anomaly,
                resume_state=resume_metrics,
                on_batch_end=save_batch,
                start_batch=start_batch,
                total_batches=total_batches,
            )
            start_batch, resume_metrics = 0, None
            validation = evaluate("val")
            scheduler.step(validation["mean_mae"])
            improved = validation["mean_mae"] < best
            stale = 0 if validation["mean_mae"] < best - 1e-5 else stale + 1
            best = min(best, validation["mean_mae"])
            history.append(
                dict(
                    epoch=epoch, train=training, val=validation, lr=optimizer.param_groups[0]["lr"]
                )
            )
            rank_states = gather_rank_state(dict(rng=capture_rng(device), metrics=None), world)
            # Commit last.pt only after a newly improved best.pt is safely written.
            paths = [checkpoint_dir / "best.pt"] if improved else []
            paths.append(checkpoint_dir / "last.pt")
            collective_save(
                lambda: make_state(epoch, "epoch_complete", total_batches, rank_states),
                paths,
                rank,
                world,
                device,
            )
            if rank == 0:
                (output / "history.json").write_text(
                    json.dumps(history, indent=2), encoding="utf-8"
                )
                print(
                    f"Epoch {epoch}: train MAE={training['mean_mae']:.6f}, val MAE={validation['mean_mae']:.6f}",
                    flush=True,
                )
            if world > 1:
                dist.barrier()
            if stale >= args.patience:
                break
        possible_best = [checkpoint_dir / "best.pt"]
        if resume_path is not None:
            possible_best.append(resume_path.parent / "best.pt")
        possible_best.append(output / "best.pt")
        best_path = next((path for path in possible_best if path.is_file()), None)
        if best_path is None:
            raise FileNotFoundError(
                "Missing best.pt. Keep the validation-best checkpoint alongside last.pt when migrating."
            )
        unwrap_model(model).load_state_dict(load_pt(best_path)["model"])
        results = {"expert_names": names, "val": evaluate("val")}
        if len(splits["test"]):
            results["test"] = evaluate("test")
        if rank == 0:
            (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def export_stage1(args):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("Export with python, one process only.")
    checkpoint = load_pt(args.checkpoint)
    # Architecture is authoritative; do not accidentally export with different history.
    for key in [
        "history_hours",
        "latent_channels",
        "decoder_channels",
        "dropout",
        "temporal_channels",
    ]:
        setattr(args, key, checkpoint["architecture"][key])
    args.no_decoder_checkpoint = not checkpoint["architecture"]["checkpoint_decoders"]
    args.expected_experts = checkpoint["architecture"]["num_experts"]
    meta, splits, valid_history, static, ocean, names = inspect_data(args)
    if meta != checkpoint["architecture"] or names != checkpoint["expert_names"]:
        raise ValueError("Export data/architecture/expert order mismatch.")
    if not np.array_equal(ocean, checkpoint["ocean_mask"]):
        raise ValueError("Export ocean mask differs from training.")
    if str(Path(args.data_h5).resolve()) != checkpoint["data_h5"]:
        raise ValueError("Export requires the original training HDF5.")
    dest = Path(args.export_h5).resolve()
    if dest.exists() or dest == Path(args.data_h5).resolve():
        raise FileExistsError("Export destination must be a new file.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = WaveExpertModel(**meta).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    ids = np.flatnonzero(valid_history)
    ds = HistoryDataset(
        args.data_h5,
        ids,
        args.history_hours,
        static,
        ocean,
        np.asarray(checkpoint["mean"]),
        np.asarray(checkpoint["std"]),
        targets=False,
    )
    partial = Path(str(dest) + ".partial")
    if partial.exists():
        raise FileExistsError(f"Remove or rename previous incomplete export: {partial}")
    with h5py.File(args.data_h5, "r") as src, h5py.File(partial, "w") as out:
        source = str(Path(args.data_h5).resolve())
        reserved = {
            "wave",
            "ww3_reference",
            "stage1_z",
            "stage1_history_valid",
            "stage1_prediction_mask",
            "train_indices",
            "val_indices",
            "test_indices",
        }
        for key in src.keys():
            if key not in reserved:
                out[key] = h5py.ExternalLink(source, "/" + key)
        for key, value in src.attrs.items():
            out.attrs[key] = value
        out.attrs["wave_source"] = "learned_stage1_experts"
        out.attrs["stage1_checkpoint"] = str(Path(args.checkpoint).resolve())
        out.attrs["stage1_architecture_json"] = json.dumps(meta)
        out["ww3_reference"] = h5py.ExternalLink(source, "/wave")
        out.create_dataset("stage1_history_valid", data=valid_history)
        for split, split_ids in splits.items():
            out.create_dataset(f"{split}_indices", data=split_ids)
        n, k, h, w = src["wave"].shape
        wave = out.create_dataset(
            "wave",
            (n, k, h, w),
            dtype="f4",
            chunks=(1, 1, h, w),
            compression="lzf",
            fillvalue=np.nan,
        )
        # Forecast validity uses sea mask, never future numerical wave targets.
        predmask = out.create_dataset(
            "stage1_prediction_mask",
            (n, k, h, w),
            dtype="?",
            chunks=(1, 1, h, w),
            compression="lzf",
            fillvalue=False,
        )
        zdata = None
        if args.export_z:
            zdata = out.create_dataset(
                "stage1_z",
                (n, meta["latent_channels"], h, w),
                dtype="f4",
                chunks=(1, 1, h, w),
                compression="lzf",
                fillvalue=np.nan,
            )
        with torch.no_grad():
            for batch in stage1_tqdm(loader(ds, args), desc="Export learned experts"):
                pred, z = model(batch["wind"].to(device), batch["static"].to(device), True)
                pred = pred.cpu().numpy()
                z = z.cpu().numpy() if zdata is not None else None
                for b, i in enumerate(batch["index"].tolist()):
                    valid = np.broadcast_to(ocean[None], (k, h, w))
                    wave[i] = np.where(valid, pred[b], np.nan)
                    predmask[i] = valid
                    if zdata is not None:
                        zdata[i] = np.where(ocean[None], z[b], np.nan)
    os.replace(partial, dest)
    print(f"Export complete: {dest}")


def parse_stage1_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("command", choices=["train", "export"])
    p.add_argument(
        "--data-h5",
        default="./global_wave_dataset_base_1h_0p5deg/global_wave_dataset_base_1h_0p5deg.h5",
    )
    p.add_argument(
        "--static-npz",
        default="./global_wave_dataset_base_1h_0p5deg/static_etopo5_3features_global_h5grid_0p5deg.npz",
    )
    p.add_argument("--history-hours", type=int, default=241)
    p.add_argument("--expected-experts", type=int, default=21)
    p.add_argument("--latent-channels", type=int, default=48)
    p.add_argument("--decoder-channels", type=int, default=32)
    p.add_argument("--temporal-channels", type=int, default=16)
    p.add_argument("--no-decoder-checkpoint", action="store_true")
    p.add_argument("--area-weighted", action="store_true")
    p.add_argument("--lambda-anomaly", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per GPU; 10-day global windows consume substantial memory.",
    )
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--amp", action="store_true", help="Opt-in CUDA float16 autocast.")
    p.add_argument(
        "--recompute-wind-stats",
        action="store_true",
        help="Ignore matching wind statistics cache and recompute it.",
    )
    p.add_argument("--ocean-mask-key", default="", help="HDF5 [H,W] binary mask, 1=ocean.")
    p.add_argument("--output", default="./outputs_wave_experts_stage1_v2")
    p.add_argument(
        "--resume", default="", help="Checkpoint path, or auto for latest saved batch/epoch."
    )
    p.add_argument("--save-every-batches", type=int, default=25)
    p.add_argument(
        "--checkpoint-dir",
        default="",
        help="Optional reliable local directory for all .pt checkpoints.",
    )
    p.add_argument(
        "--allow-data-relocation",
        action="store_true",
        help="Declare an identical dataset copy at a changed --data-h5 path.",
    )
    p.add_argument("--checkpoint", default="./outputs_wave_experts_stage1_v2/best.pt")
    p.add_argument(
        "--export-h5",
        default="./global_wave_dataset_base_1h_0p5deg/global_wave_learned_experts_1h_0p5deg.h5",
    )
    p.add_argument(
        "--export-z",
        action="store_true",
        help="Large optional latent dataset for redesigned Stage 2/3.",
    )
    args = p.parse_args()
    if (
        min(
            args.history_hours, args.batch_size, args.epochs, args.patience, args.save_every_batches
        )
        < 1
        or args.workers < 0
    ):
        p.error("history/batch/epochs/patience must be positive, workers nonnegative.")
    if (
        args.latent_channels < 4
        or args.latent_channels % 4
        or min(args.decoder_channels, args.temporal_channels) < 1
        or args.lambda_anomaly < 0
    ):
        p.error("latent channels must be divisible by 4; anomaly weight must be nonnegative.")
    return args


def main(argv=None):
    """CLI for Stage-1 expert training and export."""
    if argv is not None:
        old = sys.argv
        sys.argv = [old[0], *argv]
    else:
        old = None
    try:
        args = parse_stage1_args()
        if args.command == "train":
            train_stage1(args)
        elif args.command == "export":
            export_stage1(args)
    finally:
        if old is not None:
            sys.argv = old


if __name__ == "__main__":
    main()
