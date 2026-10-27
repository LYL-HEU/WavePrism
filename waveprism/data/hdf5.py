from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm as stage1_tqdm

from ..models.factory import get_stage1_context


def read_float(dataset, selection):
    """Honor explicit finite fill sentinels; never interpret HDF5 default zero as missing."""
    array = np.asarray(dataset[selection], dtype=np.float32)
    for key in ["_FillValue", "missing_value"]:
        if key in dataset.attrs:
            for fill in np.asarray(dataset.attrs[key]).reshape(-1):
                array[array == np.float32(fill)] = np.nan
    return array


def ocean_from_wave(reference):
    """User rule: all-zero expert cells are land; any finite positive expert is ocean."""
    return (np.isfinite(reference) & (reference > 0)).any(axis=0)


class DayBlockSampler(Sampler):
    """Shuffle calendar-day blocks, read nearby windows sequentially per rank."""

    def __init__(self, dataset, world, rank, batch, seed):
        self.world, self.rank, self.batch, self.seed = world, rank, batch, seed
        self.epoch = 0
        self.start_batch = 0
        days = dataset.times[dataset.ids] // 86_400_000_000_000
        cuts = np.flatnonzero(np.diff(days)) + 1
        self.blocks = np.split(np.arange(len(dataset)), cuts)
        self.total = len(dataset) // (world * batch) * (world * batch)
        if not self.total:
            raise ValueError("Not enough train targets for one global batch.")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_start_batch(self, batch_index):
        total_batches = self.total // (self.world * self.batch)
        if batch_index < 0 or batch_index > total_batches:
            raise ValueError("Saved batch cursor is outside this epoch.")
        self.start_batch = int(batch_index)

    def __len__(self):
        return self.total // self.world - self.start_batch * self.batch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.blocks))
        ids = np.concatenate([self.blocks[i] for i in order])[: self.total]
        ids = ids.reshape(-1, self.world, self.batch)[:, self.rank].reshape(-1)
        return iter(ids[self.start_batch * self.batch :].tolist())


def inspect_data(args, ocean_override=None):
    with h5py.File(args.data_h5, "r") as f:
        for key in ["wind", "time", "lat", "lon", "train_indices", "val_indices", "test_indices"]:
            if key not in f:
                raise KeyError(f"Missing HDF5 key: {key}")
        n, c, h, w = f["wind"].shape
        k = args.expected_experts
        if ocean_override is None:
            nw, k, hw, ww = f["wave"].shape
            if (nw, hw, ww) != (n, h, w) or k != args.expected_experts:
                raise ValueError("Wave targets must match wind grid and expected expert count.")
        if f["lat"].shape != (h,) or f["lon"].shape != (w,):
            raise ValueError("Expected 1-D lat/lon coordinates.")
        lat = np.asarray(f["lat"][:], dtype=np.float64)
        if (
            not np.isfinite(lat).all()
            or np.any(np.abs(lat) > 90)
            or not (np.all(np.diff(lat) > 0) or np.all(np.diff(lat) < 0))
        ):
            raise ValueError("lat must be finite, monotonic, in [-90,90].")
        lon = np.asarray(f["lon"][:], dtype=np.float64)
        if (
            len(lon) < 2
            or not np.allclose(np.diff(lon), np.diff(lon)[0])
            or not np.isclose(abs(lon[1] - lon[0]) * w, 360)
        ):
            raise ValueError("Circular longitude requires a complete uniform global grid.")
        # Current project stores Unix nanoseconds. Never infer time units silently.
        times = np.asarray(f["time"][:], dtype=np.int64)
        if times.shape != (n,) or np.any(np.diff(times) <= 0):
            raise ValueError("time must be strictly increasing Unix nanoseconds.")
        gaps = np.diff(times) != 3_600_000_000_000
        prefix = np.concatenate([[0], np.cumsum(gaps)])
        valid_history = np.zeros(n, dtype=bool)
        idx = np.arange(args.history_hours - 1, n)
        valid_history[idx] = prefix[idx] == prefix[idx - args.history_hours + 1]
        splits = {}
        all_ids = []
        for split in ["train", "val", "test"]:
            ids = np.asarray(f[f"{split}_indices"][:], dtype=np.int64)
            if np.any(ids < 0) or np.any(ids >= n) or len(np.unique(ids)) != len(ids):
                raise ValueError(f"Invalid or repeated {split} indices.")
            all_ids.append(ids)
            splits[split] = np.sort(ids[valid_history[ids]])
            print(f"{split}: {len(ids)} targets -> {len(splits[split])} complete histories")
        if len(np.unique(np.concatenate(all_ids))) != sum(map(len, all_ids)):
            raise ValueError("train/val/test target indices overlap.")
        if not len(splits["train"]) or not len(splits["val"]):
            raise ValueError("Need nonempty train and val after history filtering.")
        previous = None
        for split in ["train", "val", "test"]:
            if len(splits[split]):
                if previous is not None and splits[split][0] <= previous:
                    raise ValueError(
                        "Require chronological train < val < test; random hourly splits are not supported."
                    )
                previous = splits[split][-1]
        names = (
            [x.decode() if isinstance(x, bytes) else str(x) for x in f["expert_names"][:]]
            if "expert_names" in f
            else [f"T{i + 1}" for i in range(k)]
        )
        if len(names) != k or len(set(names)) != k:
            raise ValueError("Expert names must be unique and match wave channels.")
        ocean = np.asarray(ocean_override, dtype=bool) if ocean_override is not None else None
        if ocean is None and args.ocean_mask_key:
            if args.ocean_mask_key not in f:
                raise KeyError(f"No HDF5 ocean mask {args.ocean_mask_key!r}")
            ocean = np.asarray(f[args.ocean_mask_key][:]) > 0
        elif ocean is None and "ocean_mask" in f:
            ocean = np.asarray(f["ocean_mask"][:]) > 0
    with np.load(args.static_npz) as s:
        static = np.asarray(s["static"], dtype=np.float32)
        for coord, expected in [("lat", lat), ("lon", lon)]:
            if not np.allclose(s[coord], expected):
                raise ValueError(f"Static NPZ needs matching {coord} coordinates.")
        if ocean is None and "ocean_mask" in s:
            ocean = np.asarray(s["ocean_mask"]) > 0
    if static.ndim != 3 or static.shape[-2:] != (h, w):
        raise ValueError("static must have shape [S,H,W].")
    if ocean is None:
        # Infer from numerical wave values, never from sparse observation target_mask.
        with h5py.File(args.data_h5, "r") as f:
            reference = read_float(f["wave"], int(splits["train"][0]))
        ocean = ocean_from_wave(reference)
        print(
            f"Auto ocean mask from WW3 at train index {int(splits['train'][0])}: "
            f"ocean={int(ocean.sum())}, land/invalid={int((~ocean).sum())}; Hs=0 excluded."
        )
    if ocean.shape != (h, w) or not ocean.any():
        raise ValueError("Ocean mask must be nonempty [H,W], 1=ocean.")
    if not np.isfinite(static[:, ocean]).all():
        raise ValueError("Static fields contain nonfinite values over ocean.")
    position = np.stack(
        [
            np.broadcast_to(np.sin(np.deg2rad(lat))[:, None], (h, w)),
            np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (h, w)),
            np.broadcast_to(np.sin(np.deg2rad(lon))[None, :], (h, w)),
            np.broadcast_to(np.cos(np.deg2rad(lon))[None, :], (h, w)),
            np.broadcast_to((lat / 90.0)[:, None], (h, w)),
        ]
    ).astype(np.float32)
    static = np.concatenate(
        [
            np.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0),
            position,
            ocean[None].astype(np.float32),
        ],
        0,
    )
    meta = dict(
        history_hours=args.history_hours,
        wind_channels=c,
        static_channels=static.shape[0],
        num_experts=k,
        latent_channels=args.latent_channels,
        decoder_channels=args.decoder_channels,
        dropout=args.dropout,
        temporal_channels=args.temporal_channels,
        checkpoint_decoders=not args.no_decoder_checkpoint,
        version=2,
    )
    return (
        meta,
        splits,
        valid_history,
        np.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0),
        ocean,
        names,
    )


def wind_stats(path, train_ids, ocean):
    # Only training target-time winds, no validation/test winds or wave/observations.
    with h5py.File(path, "r") as f:
        c = f["wind"].shape[1]
        sums, squares, counts = [np.zeros(c, dtype=np.float64) for _ in range(3)]
        for i in stage1_tqdm(train_ids, desc="Train-only wind normalization"):
            wind = read_float(f["wind"], int(i)).astype(np.float64)[:, ocean]
            finite = np.isfinite(wind)
            safe = np.where(finite, wind, 0.0)
            sums += safe.sum(axis=1)
            squares += (safe * safe).sum(axis=1)
            counts += finite.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("Wind channel has no finite training samples.")
    mean = sums / counts
    std = np.sqrt(np.maximum(squares / counts - mean**2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def cached_wind_stats(path, train_ids, ocean, cache_path, channels, recompute=False):
    """Rank0 only. Save before network setup/training so later failures retain stats."""
    source = Path(path).resolve()
    cache_path = Path(cache_path)
    info = source.stat()
    signature = (str(source), info.st_size, info.st_mtime_ns)
    if cache_path.exists() and not recompute:
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                same_source = (
                    str(data["source_path"].item()),
                    int(data["source_size"].item()),
                    int(data["source_mtime_ns"].item()),
                ) == signature
                same_split = np.array_equal(data["train_indices"], train_ids)
                same_mask = np.array_equal(data["ocean_mask"], ocean)
                mean, std = data["mean"].copy(), data["std"].copy()
                good_stats = (
                    mean.shape == std.shape == (channels,)
                    and np.isfinite(mean).all()
                    and np.isfinite(std).all()
                    and (std > 0).all()
                )
            if same_source and same_split and same_mask and good_stats:
                print(f"Loaded train-only wind normalization cache: {cache_path}")
                return mean.astype(np.float32), std.astype(np.float32)
            print("Wind statistics cache differs from current inputs; recomputing.")
        except (OSError, ValueError, KeyError) as exc:
            print(f"Wind statistics cache could not be used ({exc}); recomputing.")
    mean, std = wind_stats(path, train_ids, ocean)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(cache_path) + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            mean=mean,
            std=std,
            train_indices=train_ids,
            ocean_mask=ocean,
            source_path=signature[0],
            source_size=np.int64(signature[1]),
            source_mtime_ns=np.int64(signature[2]),
        )
    os.replace(temporary, cache_path)
    print(f"Saved train-only wind normalization cache: {cache_path}")
    return mean, std


class HistoryDataset(Dataset):
    def __init__(self, path, ids, history, static, ocean, mean, std, targets=True):
        self.path, self.ids, self.history = path, np.asarray(ids), history
        self.static, self.ocean = static, ocean
        self.mean = mean.reshape(1, -1, 1, 1)
        self.std = std.reshape(1, -1, 1, 1)
        self.targets, self.file = targets, None
        self.cached_i, self.cached_wind = None, None
        with h5py.File(path, "r") as f:
            self.times = np.asarray(f["time"][:], dtype=np.int64)
            lat = np.asarray(f["lat"][:], dtype=np.float32)
        self.area = np.broadcast_to(
            np.maximum(np.cos(np.deg2rad(lat)), 0.0)[:, None], ocean.shape
        ).copy()

    def __len__(self):
        return len(self.ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["file"] = None
        state["cached_i"], state["cached_wind"] = None, None
        return state

    def __getitem__(self, item):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        i = int(self.ids[item])
        shift = i - self.cached_i if self.cached_i is not None else 0
        if self.cached_wind is not None and 0 < shift <= min(24, self.history - 1):
            new = read_float(self.file["wind"], slice(self.cached_i + 1, i + 1))
            wind = np.concatenate([self.cached_wind[shift:], new], axis=0)
        else:
            wind = read_float(self.file["wind"], slice(i - self.history + 1, i + 1))
        self.cached_i, self.cached_wind = i, wind
        if not np.isfinite(wind[:, :, self.ocean]).all():
            raise ValueError(f"Nonfinite ocean wind in history ending at {i}; repair input first.")
        wind = np.nan_to_num((wind - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0)
        out = {
            "wind": torch.from_numpy(wind),
            "static": torch.from_numpy(self.static),
            "index": i,
            "area": torch.from_numpy(self.area),
        }
        if self.targets:
            target = read_float(self.file["wave"], i)
            # Explicit user policy: zero-valued wave targets do not contribute to loss/metrics.
            valid = np.isfinite(target) & (target > 0) & self.ocean[None]
            out.update(
                target=torch.from_numpy(np.where(valid, target, 0.0)), valid=torch.from_numpy(valid)
            )
        return out


def loader(dataset, args, sampler=None):
    # A dedicated generator prevents iterator creation from consuming dropout RNG
    # when restarting midway through an epoch. This dataset has no random transforms.
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        generator=generator,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )


class GlobalH5WaveDataset(Dataset):
    """Same selector interface; x_in now contains the full Stage-1 wind history."""

    def __init__(
        self,
        h5_path,
        split,
        normalize_input=True,
        input_mean=None,
        input_std=None,
        static_npz=None,
        use_position_features=True,
    ):
        context = get_stage1_context()
        if (
            not normalize_input
            or not np.array_equal(input_mean, context["mean"])
            or not np.array_equal(input_std, context["std"])
        ):
            raise ValueError("Must use the pretrained Stage-1 normalization unchanged.")
        self.h5_path, self.split = str(h5_path), split
        self.indices = context["splits"][split]
        self.ids = self.indices  # Stage-1 day-block sampler interface
        self.history_data = HistoryDataset(
            h5_path,
            self.indices,
            context["meta"]["history_hours"],
            context["geometry"],
            context["ocean"],
            context["mean"],
            context["std"],
            targets=False,
        )
        self.times = self.history_data.times
        self.ocean = context["ocean"]
        self.h, self.w = self.ocean.shape
        if static_npz is None:
            raise ValueError("static_npz is required for the frozen encoder.")
        with np.load(static_npz) as data:
            self.static = np.nan_to_num(
                np.asarray(data["static"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
        self.position = context["geometry"][-6:-2].copy() if use_position_features else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.history_data[idx]
        f = self.history_data.file
        i = int(self.indices[idx])
        y = read_float(f["satellite"], i)[None]
        mask = np.asarray(f["target_mask"][i], dtype=bool)[None]
        mask &= self.ocean[None] & np.isfinite(y)
        out = dict(
            x_in=sample["wind"],
            y_sat=torch.from_numpy(np.ascontiguousarray(y)),
            mask=torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32))),
            static=torch.from_numpy(self.static),
        )
        if self.position is not None:
            out["position"] = torch.from_numpy(self.position)
        return out
