from __future__ import annotations

import json
import math
from datetime import timedelta

import numpy as np

from ..inference.common import array_hash
from .ifs_wind import LEADS, iso


def geometry_from_static(stage1, static_path):
    arch = stage1["architecture"]
    ocean = np.asarray(stage1["ocean_mask"], dtype=bool)
    if arch["wind_channels"] != 2 or int(arch["history_hours"]) < 2:
        raise ValueError("Operational inference requires an hourly u10/v10 model.")
    with np.load(static_path) as data:
        lat = np.asarray(data["lat"], dtype=np.float64)
        lon = np.asarray(data["lon"], dtype=np.float64)
        static = np.asarray(data["static"], dtype="f4")
    if lat.ndim != 1 or lon.ndim != 1 or ocean.shape != (len(lat), len(lon)):
        raise ValueError("Static grid differs from the model ocean mask.")
    if (
        static.ndim != 3
        or static.shape[1:] != ocean.shape
        or static.shape[0] + 6 != arch["static_channels"]
    ):
        raise ValueError("Static channels or grid differ from Stage 1.")
    if (
        not np.isfinite(static[:, ocean]).all()
        or not np.isfinite(lat).all()
        or not np.isfinite(lon).all()
    ):
        raise ValueError("Static features or coordinates contain nonfinite values.")
    if (
        len(lon) < 2
        or not np.allclose(np.diff(lon), np.diff(lon)[0])
        or not np.isclose(abs(lon[1] - lon[0]) * len(lon), 360)
    ):
        raise ValueError("Model longitudes must cover a complete uniform 360-degree grid.")
    h, w = ocean.shape
    position = np.stack(
        [
            np.broadcast_to(np.sin(np.deg2rad(lat))[:, None], (h, w)),
            np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (h, w)),
            np.broadcast_to(np.sin(np.deg2rad(lon))[None], (h, w)),
            np.broadcast_to(np.cos(np.deg2rad(lon))[None], (h, w)),
            np.broadcast_to((lat / 90.0)[:, None], (h, w)),
        ]
    ).astype("f4")
    static = np.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0)
    geometry = np.concatenate([static, position, ocean[None].astype("f4")])
    return dict(
        lat=lat, lon=lon, static=static, position=position[:4], ocean=ocean, geometry=geometry
    )


def prepare_forcing(init, path, analysis_index, reader, history_hours, args):
    lookback = int(math.ceil((history_hours - 1) / 6.0)) * 6
    offsets = list(range(-lookback, 0, 6))
    required = [init + timedelta(hours=value) for value in offsets]
    if args.t0_source == "analysis":
        required.append(init)
    missing = [stamp for stamp in required if stamp not in analysis_index]
    if missing:
        raise FileNotFoundError(
            f"{iso(init)}: missing {len(missing)} analysis files; first: 10m_wind_{missing[0]:%Y%m%d_%H%M}.nc"
        )
    future, interpretation = reader.forecast(path, init)
    anchors = np.empty((len(offsets) + len(LEADS), 2, *reader.meta["ocean"].shape), dtype="f4")
    files = []
    for row, stamp in enumerate(required[: len(offsets)]):
        file = analysis_index[stamp]
        files.append(file)
        anchors[row] = reader.analysis(file, stamp)
    if args.t0_source == "analysis":
        files.append(analysis_index[init])
        future[0] = reader.analysis(analysis_index[init], init)
    anchors[len(offsets) :] = future
    return dict(
        anchors=anchors,
        start_hour=-lookback,
        analysis_files=files,
        forecast_file=path,
        future=future,
        interpretation=interpretation,
    )


class HourlyWindows:
    """Interpolate u/v components, normalize once, and reuse 6-hour window overlap."""

    def __init__(self, forcing, stage1):
        self.forcing = forcing
        self.history = int(stage1["architecture"]["history_hours"])
        self.mean = np.asarray(stage1["mean"], dtype="f4")[:, None, None]
        self.std = np.asarray(stage1["std"], dtype="f4")[:, None, None]
        self.cached, self.last_hour = None, None

    def hour(self, hour):
        offset = int(hour) - self.forcing["start_hour"]
        lower, remainder = divmod(offset, 6)
        anchors = self.forcing["anchors"]
        if lower < 0 or lower >= len(anchors) or (remainder and lower + 1 >= len(anchors)):
            raise ValueError(
                "Requested wind hour lies outside forcing anchors; extrapolation is not allowed."
            )
        value = (
            anchors[lower]
            if remainder == 0
            else (
                anchors[lower] * (np.float32(1) - np.float32(remainder / 6.0))
                + anchors[lower + 1] * np.float32(remainder / 6.0)
            )
        )
        return np.nan_to_num((value - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0)

    def window(self, lead):
        shift = int(lead) - self.last_hour if self.last_hour is not None else self.history
        output = np.empty((self.history, *self.forcing["anchors"].shape[1:]), dtype="f4")
        if self.cached is not None and 0 < shift < self.history:
            output[:-shift] = self.cached[shift:]
            first = self.history - shift
        else:
            first = 0
        for row in range(first, self.history):
            output[row] = self.hour(int(lead) - self.history + 1 + row)
        self.cached, self.last_hour = output, int(lead)
        return output


def input_signature(init, forcing, args, hashes, meta, expert_ids):
    files = [forcing["forecast_file"]] + forcing["analysis_files"]
    stamps = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    return json.dumps(
        dict(
            version=2,
            init=iso(init),
            inputs=stamps,
            hashes=hashes,
            geometry=array_hash(meta["geometry"]),
            stages=args.stages,
            experts=expert_ids,
            t0_source=args.t0_source,
            leads=LEADS.tolist(),
            interpolation="componentwise linear 6h to 1h",
            wind_selectors=[
                args.u_var,
                args.v_var,
                args.lat_var,
                args.lon_var,
                args.step_var,
                args.step_unit,
            ],
            step_interpretation=forcing["interpretation"],
            save_z=args.save_z,
            save_prob=args.save_selection_probabilities,
            save_wind=args.save_wind,
            amp=args.amp,
            member=args.member,
            compression=args.compression_level,
        ),
        sort_keys=True,
    )
