from __future__ import annotations

import hashlib
import json
import math
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def setup():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-process training requires CUDA/NCCL.")
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
    return rank, world, torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")


def reduce_sum(t):
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def make_grad_scaler(enabled):
    return torch.amp.GradScaler("cuda", enabled=enabled)


def atomic_save(obj, path):
    """A failed temporary write leaves the previous checkpoint intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(obj, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass


def dataset_signature(path, static, ocean, names):
    digest = hashlib.sha256()
    with h5py.File(path, "r") as file:
        digest.update(str(tuple(file["wind"].shape)).encode())
        digest.update(str(tuple(file["wave"].shape)).encode())
        for key in ["time", "lat", "lon", "train_indices", "val_indices", "test_indices"]:
            array = np.ascontiguousarray(file[key][:])
            digest.update(key.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    digest.update(json.dumps(names).encode())
    digest.update(np.ascontiguousarray(static).tobytes())
    digest.update(np.ascontiguousarray(ocean).tobytes())
    return digest.hexdigest()


def capture_rng(device):
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    )


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def gather_rank_state(local_state, world):
    if world == 1:
        return [local_state]
    states = [None] * world
    dist.all_gather_object(states, local_state)
    return states


def checkpoint_order(checkpoint):
    # At the same epoch, a completed-validation save wins over an in-epoch save.
    complete = checkpoint.get("phase", "epoch_complete") == "epoch_complete"
    return (int(checkpoint["epoch"]), int(complete), int(checkpoint.get("next_batch", 0)))


def resolve_resume(value, checkpoint_dir, output):
    if not value:
        return None
    if value != "auto":
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return path
    candidates = []
    for directory in dict.fromkeys([Path(checkpoint_dir).resolve(), Path(output).resolve()]):
        for name in ["last.pt", "last_batch.pt"]:
            path = directory / name
            if path.is_file():
                candidates.append(path)
    usable = []
    for path in candidates:
        try:
            state = load_pt(path)
            if not all(key in state for key in ["model", "optimizer", "architecture", "epoch"]):
                raise ValueError("Not a complete training checkpoint.")
            usable.append((checkpoint_order(state), path))
        except Exception as exc:
            print(f"Cannot use checkpoint {path}: {exc}", flush=True)
    if not usable:
        raise FileNotFoundError(
            "--resume auto found no readable training checkpoint. "
            "Use no --resume only when starting a NEW run."
        )
    return max(usable, key=lambda item: item[0])[1]


def collective_save(state_factory, paths, rank, world, device):
    """Communicate rank0 write failure, rather than leaving peers in a barrier."""
    failure = None
    if rank == 0:
        try:
            state = state_factory()
            for path in paths:
                atomic_save(state, path)
        except Exception as exc:
            failure = exc
            print(f"Checkpoint save failed; previous committed file retained: {exc}", flush=True)
    status = torch.tensor([0 if failure is not None else 1], device=device, dtype=torch.int32)
    if world > 1:
        dist.broadcast(status, src=0)
    if int(status.item()) != 1:
        raise OSError(
            "Checkpoint write failed; inspect output storage. Previous saved progress is retained."
        ) from failure


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor_to_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def format_array(arr) -> str:
    arr = np.asarray(arr, dtype=np.float64)
    return "[" + ", ".join([f"{x:.4f}" for x in arr]) + "]"


def torch_load_safe(path: str, map_location):
    return torch.load(path, map_location=map_location, weights_only=False)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def setup_distributed():
    distributed = (
        "RANK" in os.environ
        and "LOCAL_RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )

    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=6))
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return rank, local_rank, world_size, device, distributed


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_rank0(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


class PointMetricTracker:
    def __init__(self):
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.bias_sum = 0.0
        self.rel_sum = 0.0
        self.abs_rel_sum = 0.0
        self.n = 0.0
        self.n_rel = 0.0

    def update(self, pred: torch.Tensor, obs: torch.Tensor, valid: torch.Tensor):
        keep = (valid > 0.5) & torch.isfinite(pred) & torch.isfinite(obs)
        pred, obs = pred.detach()[keep], obs.detach()[keep]
        diff = pred - obs
        self.abs_sum += diff.abs().sum().item()
        self.sq_sum += diff.square().sum().item()
        self.bias_sum += diff.sum().item()
        self.n += diff.numel()
        relative = obs.abs() > 1e-6
        rel = diff[relative] / obs[relative]
        self.rel_sum += rel.sum().item()
        self.abs_rel_sum += rel.abs().sum().item()
        self.n_rel += rel.numel()

    def compute(self) -> Dict[str, float]:
        n = max(self.n, 1.0)
        n_rel = max(self.n_rel, 1.0)

        return {
            "N": int(self.n),
            "bias": self.bias_sum / n,
            "mae": self.abs_sum / n,
            "rmse": math.sqrt(self.sq_sum / n),
            "mre_percent": self.rel_sum / n_rel * 100.0,
            "mare_percent": self.abs_rel_sum / n_rel * 100.0,
        }


class ProbMetricTracker:
    def __init__(self):
        self.nll_sum = 0.0
        self.crps_sum = 0.0
        self.n = 0.0

        self.cover90_sum = 0.0
        self.width90_sum = 0.0

        self.cover95_sum = 0.0
        self.width95_sum = 0.0

        self.sigma_sum = 0.0
        self.sigma_min = float("inf")
        self.sigma_max = 0.0

    @staticmethod
    def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _normal_pdf(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)

    def update(self, mu: torch.Tensor, sigma: torch.Tensor, obs: torch.Tensor, valid: torch.Tensor):
        keep = (
            (valid > 0.5)
            & torch.isfinite(mu)
            & torch.isfinite(sigma)
            & (sigma > 0)
            & torch.isfinite(obs)
        )
        mu, sigma, obs = mu.detach()[keep], sigma.detach()[keep].clamp_min(1e-6), obs.detach()[keep]
        z = (obs - mu) / sigma
        nll = 0.5 * z.square() + sigma.log() + 0.5 * math.log(2.0 * math.pi)
        crps = sigma * (
            z * (2.0 * self._normal_cdf(z) - 1.0)
            + 2.0 * self._normal_pdf(z)
            - 1.0 / math.sqrt(math.pi)
        )
        self.nll_sum += nll.sum().item()
        self.crps_sum += crps.sum().item()
        self.n += z.numel()
        for level, zq in [(90, 1.6448536269514722), (95, 1.959963984540054)]:
            covered = ((obs >= mu - zq * sigma) & (obs <= mu + zq * sigma)).sum().item()
            width = (2 * zq * sigma).sum().item()
            if level == 90:
                self.cover90_sum += covered
                self.width90_sum += width
            else:
                self.cover95_sum += covered
                self.width95_sum += width
        if sigma.numel():
            self.sigma_sum += sigma.sum().item()
            self.sigma_min = min(self.sigma_min, sigma.min().item())
            self.sigma_max = max(self.sigma_max, sigma.max().item())

    def compute(self) -> Dict[str, float]:
        n = max(self.n, 1.0)

        return {
            "nll": self.nll_sum / n,
            "crps": self.crps_sum / n,
            "picp90": self.cover90_sum / n,
            "mpiw90": self.width90_sum / n,
            "picp95": self.cover95_sum / n,
            "mpiw95": self.width95_sum / n,
            "sigma_mean": self.sigma_sum / n,
            "sigma_min": self.sigma_min if self.sigma_min < float("inf") else float("nan"),
            "sigma_max": self.sigma_max,
        }


def _init_metric_agg(metrics: Dict[str, Any]):
    agg = {}
    for k, v in metrics.items():
        if isinstance(v, list):
            agg[k] = np.zeros(len(v), dtype=np.float64)
        else:
            agg[k] = 0.0
    return agg


def _update_metric_agg(agg: Dict[str, Any], metrics: Dict[str, Any]):
    for k, v in metrics.items():
        if isinstance(v, list):
            agg[k] += np.asarray(v, dtype=np.float64)
        else:
            agg[k] += float(v)


def _reduce_and_finalize_agg(agg, n_batches, device, distributed_reduce: bool):
    if agg is None:
        return {}

    result = {}
    for k, v in agg.items():
        if isinstance(v, np.ndarray):
            t = torch.as_tensor(v, dtype=torch.float64, device=device)
            if distributed_reduce and dist.is_initialized():
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            result[k] = t.cpu().numpy()
        else:
            t = torch.tensor(float(v), dtype=torch.float64, device=device)
            if distributed_reduce and dist.is_initialized():
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            result[k] = float(t.item())

    nb = torch.tensor(float(n_batches), dtype=torch.float64, device=device)
    if distributed_reduce and dist.is_initialized():
        dist.all_reduce(nb, op=dist.ReduceOp.SUM)
    denom = max(float(nb.item()), 1.0)

    out = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            out[k] = (v / denom).tolist()
        else:
            out[k] = v / denom
    return out


def broadcast_scalar(value: float, device, src: int = 0) -> float:
    t = torch.tensor([float(value)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(t, src=src)
    return float(t.item())


def broadcast_bool(value: bool, device, src: int = 0) -> bool:
    t = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(t, src=src)
    return bool(int(t.item()))


def reduce_weighted_sums(
    sums: Dict[str, float], weight: float, device, distributed: bool
) -> Dict[str, float]:
    """Reduce metric sums weighted by valid-pixel count across ranks."""
    keys = list(sums.keys())
    vals = [float(sums[k]) for k in keys] + [float(weight)]
    t = torch.tensor(vals, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    total_weight = max(float(t[-1].item()), 1.0)
    return {k: float(t[i].item()) / total_weight for i, k in enumerate(keys)}
