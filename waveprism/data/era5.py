from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import netCDF4
import numpy as np
import torch
from torch.utils.data import Dataset

from ..inference.common import time_text, utc_ns
from .forcing import geometry_from_static


def _find_nc_variable(ds, explicit, candidates, role):
    """Find a NetCDF variable case-insensitively, preferring an explicit name."""
    if explicit:
        if explicit in ds.variables:
            return explicit
        lower = {name.lower(): name for name in ds.variables}
        if explicit.lower() in lower:
            return lower[explicit.lower()]
        raise KeyError(f"Missing explicit {role} variable {explicit!r} in {ds.filepath()}.")
    lower = {name.lower(): name for name in ds.variables}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise KeyError(f"Cannot identify {role} in {ds.filepath()}; variables: {list(ds.variables)}")


def _decode_nc_time(time_var):
    """Decode a 1-D NetCDF time coordinate into UTC datetime64[ns]."""
    raw = np.asarray(time_var[:])
    if raw.ndim != 1:
        raise ValueError(
            f"Time coordinate {time_var.name} must be one-dimensional, found {raw.shape}."
        )
    if np.issubdtype(raw.dtype, np.datetime64):
        result = raw.astype("datetime64[ns]")
    elif raw.dtype.kind in ("S", "U", "O"):
        result = np.asarray(
            [
                np.datetime64(
                    str(x.decode() if isinstance(x, bytes) else x).replace(" ", "T"), "ns"
                )
                for x in raw
            ],
            dtype="datetime64[ns]",
        )
    else:
        units = getattr(time_var, "units", None)
        if not units:
            raise ValueError(f"Numeric time coordinate {time_var.name} has no units.")
        calendar = getattr(time_var, "calendar", "standard")
        dates = netCDF4.num2date(
            raw, units=units, calendar=calendar, only_use_cftime_datetimes=False
        )
        values = []
        for value in np.asarray(dates).reshape(-1):
            # ERA5 uses Gregorian UTC dates; convert without a local-time shift.
            text = (
                value.strftime("%Y-%m-%dT%H:%M:%S.%f") if hasattr(value, "strftime") else str(value)
            )
            values.append(np.datetime64(text.replace(" ", "T"), "ns"))
        result = np.asarray(values, dtype="datetime64[ns]")
    if np.isnat(result).any():
        raise ValueError(f"Time coordinate {time_var.name} contains NaT.")
    return result


def _coordinate_indices(source, target, longitude=False, tolerance=1.0e-4):
    source = np.asarray(source, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("NetCDF coordinates contain nonfinite values.")
    if longitude:
        source_cmp = np.mod(source, 360.0)
        target_cmp = np.mod(target, 360.0)
        distance = np.abs(((target_cmp[:, None] - source_cmp[None, :] + 180.0) % 360.0) - 180.0)
    else:
        distance = np.abs(target[:, None] - source[None, :])
    indices = np.argmin(distance, axis=1)
    errors = distance[np.arange(len(target)), indices]
    if np.any(errors > tolerance):
        j = int(np.argmax(errors))
        kind = "longitude" if longitude else "latitude"
        raise ValueError(
            f"NetCDF {kind} does not match the model grid: target={target[j]}, nearest error={errors[j]:.6g} degrees."
        )
    if len(np.unique(indices)) != len(indices):
        raise ValueError(
            "NetCDF coordinate matching is not one-to-one; check for duplicate source coordinates."
        )
    return indices.astype(np.int64)


def _inspect_era5_file(path, args, target_lat, target_lon):
    """Read file metadata only; wind fields are loaded lazily during inference."""
    with netCDF4.Dataset(str(path), "r") as ds:
        time_name = _find_nc_variable(
            ds, args.time_var, ("time", "valid_time", "forecast_time", "datetime"), "time"
        )
        lat_name = _find_nc_variable(ds, args.lat_var, ("latitude", "lat"), "latitude")
        lon_name = _find_nc_variable(ds, args.lon_var, ("longitude", "lon"), "longitude")
        u_name = _find_nc_variable(ds, args.u10_var, ("u10", "10u", "u10m", "eastward_wind"), "u10")
        v_name = _find_nc_variable(
            ds, args.v10_var, ("v10", "10v", "v10m", "northward_wind"), "v10"
        )
        time_var, lat_var, lon_var = (
            ds.variables[time_name],
            ds.variables[lat_name],
            ds.variables[lon_name],
        )
        if (
            len(time_var.dimensions) != 1
            or len(lat_var.dimensions) != 1
            or len(lon_var.dimensions) != 1
        ):
            raise ValueError(
                f"{path}: time/latitude/longitude coordinates must be one-dimensional."
            )
        time_dim, lat_dim, lon_dim = (
            time_var.dimensions[0],
            lat_var.dimensions[0],
            lon_var.dimensions[0],
        )
        source_lat = np.asarray(lat_var[:], dtype=np.float64)
        source_lon = np.asarray(lon_var[:], dtype=np.float64)
        lat_indices = _coordinate_indices(source_lat, target_lat, longitude=False)
        lon_indices = _coordinate_indices(source_lon, target_lon, longitude=True)
        times = _decode_nc_time(time_var)
        if len(times) != len(ds.dimensions[time_dim]):
            raise ValueError(f"{path}: time coordinate length differs from its dimension.")
        for name in (u_name, v_name):
            var = ds.variables[name]
            if (
                time_dim not in var.dimensions
                or lat_dim not in var.dimensions
                or lon_dim not in var.dimensions
            ):
                raise ValueError(
                    f"{path}: {name} must contain time/latitude/longitude dimensions, found {var.dimensions}."
                )
        info = dict(
            path=str(Path(path).resolve()),
            time_name=time_name,
            time_dim=time_dim,
            lat_name=lat_name,
            lat_dim=lat_dim,
            lon_name=lon_name,
            lon_dim=lon_dim,
            u_name=u_name,
            v_name=v_name,
            lat_indices=lat_indices,
            lon_indices=lon_indices,
            source_lat_count=len(source_lat),
            source_lon_count=len(source_lon),
        )
    return info, times


def read_era5_metadata(args, stage1):
    """Index hourly ERA5 files and prepare model geometry without opening any HDF5 dataset."""
    metadata = geometry_from_static(stage1, args.static_npz)
    lat, lon = metadata["lat"], metadata["lon"]
    static, position = metadata["static"], metadata["position"]
    ocean, geometry = metadata["ocean"], metadata["geometry"]
    history = int(stage1["architecture"]["history_hours"])

    root = Path(args.era5_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"ERA5 input directory does not exist: {root}")
    paths = sorted(p for p in root.rglob(args.file_pattern) if p.is_file())
    excluded = {
        Path(args.output_nc).expanduser().resolve(),
        Path(str(Path(args.output_nc).expanduser().resolve()) + ".partial"),
    }
    paths = [p for p in paths if p.resolve() not in excluded]
    if not paths:
        raise FileNotFoundError(f"No NetCDF files matching {args.file_pattern!r} under {root}.")

    file_info = {}
    time_locations = {}
    manifest = []
    skipped_files = []
    for path in paths:
        try:
            info, file_times = _inspect_era5_file(path, args, lat, lon)
        except KeyError as exc:
            # Permit unrelated .nc files in the directory, but report each skipped file.
            skipped_files.append((str(path), str(exc)))
            continue
        info["times_count"] = len(file_times)
        file_info[info["path"]] = info
        stat = path.stat()
        manifest.append((info["path"], int(stat.st_size), int(stat.st_mtime_ns)))
        for local_index, time_value in enumerate(file_times):
            ns = int(time_value.astype("datetime64[ns]").astype(np.int64))
            if ns % 3_600_000_000_000 != 0:
                raise ValueError(f"{path}: time {time_value} is not a whole UTC hour.")
            if ns in time_locations:
                old_path, old_index = time_locations[ns]
                raise ValueError(
                    f"Duplicate time {time_text(np.asarray([ns], dtype=np.int64))[0]}: {old_path}[{old_index}] and {info['path']}[{local_index}]."
                )
            time_locations[ns] = (info["path"], int(local_index))
    if not time_locations:
        details = "\n".join(f"  {path}: {reason}" for path, reason in skipped_files[:10])
        raise RuntimeError(
            "No NetCDF files contain identifiable ERA5 u10/v10 and time/latitude/longitude coordinates."
            + ("\n" + details if details else "")
        )
    if skipped_files:
        print(
            f"[INFO] Skipped {len(skipped_files)} NetCDF files without the required ERA5 variables.",
            flush=True,
        )

    times = np.asarray(sorted(time_locations), dtype=np.int64)
    locations = [time_locations[int(ns)] for ns in times]
    n = len(times)
    if n > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("ERA5 times must be strictly increasing.")

    if args.year is not None:
        lower, upper = utc_ns(f"{args.year:04d}-01-01"), utc_ns(f"{args.year + 1:04d}-01-01")
        selection = f"year={args.year}"
    else:
        lower = utc_ns(args.start) if args.start else None
        upper = utc_ns(args.end) if args.end else None
        selection = f"start={args.start or 'all'};end={args.end or 'all'}"
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("--end must follow --start and is exclusive.")
    requested = np.arange(n, dtype=np.int64)
    if lower is not None:
        requested = requested[times[requested] >= lower]
    if upper is not None:
        requested = requested[times[requested] < upper]
    if not len(requested):
        raise ValueError("No ERA5 target times in the requested interval.")

    history_valid = np.zeros(n, dtype=bool)
    if n >= history:
        indices = np.arange(history - 1, n, dtype=np.int64)
        gap_prefix = np.r_[0, np.cumsum(np.diff(times) != 3_600_000_000_000)]
        history_valid[indices] = gap_prefix[indices] == gap_prefix[indices - history + 1]
    skipped = requested[~history_valid[requested]]
    if len(skipped) and args.invalid_history == "error":
        raise ValueError(
            f"{len(skipped)} targets lack a continuous {history}-hour history; first: {time_text(times[skipped[:1]])[0]}."
        )
    selected = requested[history_valid[requested]]
    if not len(selected):
        raise ValueError("No target in the requested interval has a complete continuous history.")

    source_manifest_hash = hashlib.sha256(
        json.dumps(sorted(manifest), separators=(",", ":")).encode()
    ).hexdigest()
    return dict(
        times=times,
        lat=lat,
        lon=lon,
        ocean=ocean,
        geometry=geometry,
        static=static,
        position=position[:4],
        ids=selected,
        skipped_ids=skipped,
        requested_count=len(requested),
        split=selection,
        file_info=file_info,
        locations=locations,
        source_manifest=sorted(manifest),
        source_manifest_hash=source_manifest_hash,
        source_root=str(root),
        history_hours=history,
    )


class Era5WindReader:
    """Lazy hourly wind reader with a small per-process NetCDF handle cache."""

    def __init__(self, metadata, max_open_files=4):
        self.metadata = metadata
        self.max_open_files = max(1, int(max_open_files))
        self.open_files = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["open_files"] = OrderedDict()
        return state

    def close(self):
        for ds in self.open_files.values():
            try:
                ds.close()
            except Exception:
                pass
        self.open_files.clear()

    def _dataset(self, path):
        if path in self.open_files:
            ds = self.open_files.pop(path)
            self.open_files[path] = ds
            return ds
        ds = netCDF4.Dataset(path, "r")
        self.open_files[path] = ds
        while len(self.open_files) > self.max_open_files:
            _, old = self.open_files.popitem(last=False)
            old.close()
        return ds

    @staticmethod
    def _read_field(ds, var_name, info, local_time_index):
        var = ds.variables[var_name]
        selected = []
        kept = []
        for dim in var.dimensions:
            if dim == info["time_dim"]:
                selected.append(int(local_time_index))
            elif dim == info["lat_dim"]:
                selected.append(slice(None))
                kept.append("lat")
            elif dim == info["lon_dim"]:
                selected.append(slice(None))
                kept.append("lon")
            elif dim.lower().startswith("expver") and len(ds.dimensions[dim]) > 1:
                selected.append(slice(None))
                kept.append("expver")
            elif len(ds.dimensions[dim]) == 1:
                selected.append(0)
            else:
                raise ValueError(
                    f"Unsupported extra dimension {dim}={len(ds.dimensions[dim])} in {var_name}."
                )
        if info["time_dim"] not in var.dimensions:
            raise ValueError(f"Wind variable {var_name} has no time dimension {info['time_dim']}.")
        raw = var[tuple(selected)]
        if np.ma.isMaskedArray(raw):
            array = np.asarray(np.ma.filled(raw, np.nan), dtype=np.float32)
        else:
            array = np.asarray(raw, dtype=np.float32)
        for attr in ("_FillValue", "missing_value"):
            if hasattr(var, attr):
                for fill in np.asarray(getattr(var, attr)).reshape(-1):
                    array[array == np.float32(fill)] = np.nan
        if "expver" in kept:
            order = [kept.index("lat"), kept.index("lon"), kept.index("expver")]
            array = np.transpose(array, order)
            merged = np.full(array.shape[:2], np.nan, dtype=np.float32)
            for k in range(array.shape[-1]):
                layer = array[..., k]
                valid = np.isfinite(layer)
                merged[valid] = layer[valid]
            array = merged
        else:
            if len(kept) != 2 or set(kept) != {"lat", "lon"}:
                raise ValueError(
                    f"Wind variable {var_name} is not a latitude/longitude field after selecting time/singletons: {kept}."
                )
            array = np.transpose(array, [kept.index("lat"), kept.index("lon")])
        array = array[np.ix_(info["lat_indices"], info["lon_indices"])]
        return np.asarray(array, dtype=np.float32)

    def read_wind_at(self, global_index):
        path, local_time_index = self.metadata["locations"][int(global_index)]
        info = self.metadata["file_info"][path]
        ds = self._dataset(path)
        u = self._read_field(ds, info["u_name"], info, local_time_index)
        v = self._read_field(ds, info["v_name"], info, local_time_index)
        expected = self.metadata["ocean"].shape
        if u.shape != expected or v.shape != expected:
            raise ValueError(
                f"Unexpected ERA5 shapes: u10={u.shape}, v10={v.shape}, expected={expected}."
            )
        return np.stack((u, v), axis=0)


class InferenceDataset(Dataset):
    """Read continuous hourly ERA5 histories; history can cross daily file boundaries."""

    def __init__(self, output_rows, metadata, stage1, max_open_files=4):
        self.rows = np.asarray(output_rows, dtype=np.int64)
        self.ids = metadata["ids"][self.rows]
        self.history_hours = int(stage1["architecture"]["history_hours"])
        self.metadata = metadata
        self.max_open_files = int(max_open_files)
        self.reader = Era5WindReader(metadata, max_open_files=self.max_open_files)
        self.ocean = metadata["ocean"]
        self.mean = np.asarray(stage1["mean"], dtype=np.float32).reshape(1, -1, 1, 1)
        self.std = np.asarray(stage1["std"], dtype=np.float32).reshape(1, -1, 1, 1)
        self.cached_i = None
        self.cached_wind = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["reader"] = Era5WindReader(self.metadata, max_open_files=self.max_open_files)
        state["cached_i"] = None
        state["cached_wind"] = None
        return state

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, row):
        target_index = int(self.ids[row])
        shift = target_index - self.cached_i if self.cached_i is not None else 0
        if self.cached_wind is not None and 0 < shift <= min(24, self.history_hours - 1):
            new = [self.reader.read_wind_at(i) for i in range(self.cached_i + 1, target_index + 1)]
            raw_history = np.concatenate((self.cached_wind[shift:], np.stack(new, axis=0)), axis=0)
        else:
            first = target_index - self.history_hours + 1
            raw_history = np.stack(
                [self.reader.read_wind_at(i) for i in range(first, target_index + 1)], axis=0
            )
        self.cached_i = target_index
        self.cached_wind = raw_history
        if raw_history.shape[0] != self.history_hours:
            raise RuntimeError(
                f"Unexpected history length: {raw_history.shape[0]} != {self.history_hours}."
            )
        if not np.isfinite(raw_history[:, :, self.ocean]).all():
            bad = int(
                np.size(raw_history[:, :, self.ocean])
                - np.isfinite(raw_history[:, :, self.ocean]).sum()
            )
            raise ValueError(
                f"ERA5 history at {time_text(self.metadata['times'][target_index : target_index + 1])[0]} contains {bad} nonfinite ocean winds."
            )
        normalized = (raw_history - self.mean) / self.std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        )
        return {"row": int(self.rows[row]), "wind": torch.from_numpy(normalized)}
