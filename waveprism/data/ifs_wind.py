from __future__ import annotations

import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np

from ..inference.common import utc_ns

HOUR_NS = 3600000000000

LEADS = np.arange(0, 241, 6, dtype=np.int32)

ANALYSIS_RE = re.compile(r"^10m_wind_(\d{8})_(\d{4})\.nc$", re.I)

FORECAST_RE = re.compile(r"^ifs_control_u10v10_(\d{8})_(\d{2})(?:_0p5deg)?\.nc$", re.I)

MEMBER_DIMS = {"number", "member", "members", "ensemble", "realization"}

LAT_NAMES = ("lat", "latitude")

LON_NAMES = ("lon", "longitude")

U_NAMES = ("u10", "10u", "u10m", "u_10m", "u", "10m_u_component_of_wind")

V_NAMES = ("v10", "10v", "v10m", "v_10m", "v", "10m_v_component_of_wind")


def date(value):
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError("Expected YYYYMMDD or YYYY-MM-DD: " + value)


def ns(value):
    return utc_ns(value.isoformat())


def iso(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def index_files(directory, regex, forecast=False):
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        match = regex.fullmatch(path.name)
        if not match:
            continue
        stamp = datetime.strptime("".join(match.groups()), "%Y%m%d%H" if forecast else "%Y%m%d%H%M")
        if (forecast and stamp.hour != 0) or stamp.minute != 0 or stamp.hour % 6:
            continue
        if stamp in files:
            raise ValueError(
                f"Multiple input files for the same timestamp: {files[stamp]} / {path}"
            )
        files[stamp] = path.resolve()
    return files


def variable(ds, explicit, candidates, role):
    lower = {key.lower(): key for key in ds.variables}
    for name in [explicit] if explicit else candidates:
        if name and name.lower() in lower:
            return ds.variables[lower[name.lower()]]
    raise ValueError(f"Missing {role} variable; available variables: {list(ds.variables)}")


def plain(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def datetime_values(var):
    """Return Unix ns only when a coordinate explicitly encodes absolute time."""
    values = np.ma.asarray(var[:])
    if np.ma.getmaskarray(values).any():
        raise ValueError("Missing values in time coordinate: " + var.name)
    array = np.asarray(values)
    if array.dtype.kind in "SUO":
        if array.dtype.kind == "S" and array.ndim > 1 and array.dtype.itemsize == 1:
            array = netCDF4.chartostring(array)
        try:
            return np.asarray([utc_ns(plain(x)) for x in array.reshape(-1)], dtype="i8").reshape(
                array.shape
            )
        except (ValueError, TypeError):
            return None
    units = plain(getattr(var, "units", "")).strip()
    if "since" not in units:
        return None
    calendar = plain(getattr(var, "calendar", "standard")).lower()
    if calendar not in ("standard", "gregorian", "proleptic_gregorian"):
        raise ValueError("Unsupported calendar: " + calendar)
    unit, origin = units.split("since", 1)
    scales = {
        "s": 1e9,
        "second": 1e9,
        "seconds": 1e9,
        "h": 36e11,
        "hour": 36e11,
        "hours": 36e11,
        "day": 864e11,
        "days": 864e11,
        "ns": 1.0,
        "nanoseconds": 1.0,
        "microseconds": 1e3,
        "milliseconds": 1e6,
        "minutes": 60e9,
    }
    if unit.strip().lower() not in scales:
        raise ValueError("Unsupported time units: " + units)
    origin = origin.strip().replace(" UTC", "+00:00")
    base = utc_ns(origin)
    delta = np.rint(array.astype(np.longdouble) * scales[unit.strip().lower()])
    return (delta + base).astype("i8")


def coordinate_vector(var, axis):
    if axis not in var.dimensions:
        return None
    if any(len(var.group().dimensions[d]) != 1 for d in var.dimensions if d != axis):
        return None
    return (
        (slice(None),)
        if var.ndim == 1
        else tuple(slice(None) if d == axis else 0 for d in var.dimensions)
    )


def forecast_steps(ds, u, lat_dim, lon_dim, init, args):
    candidates = [
        d
        for d in u.dimensions
        if d not in (lat_dim, lon_dim)
        and d.lower() not in MEMBER_DIMS
        and len(ds.dimensions[d]) > 1
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot identify forecast lead axis: dimensions={u.dimensions}, shape={u.shape}"
        )
    axis = candidates[0]
    size = len(ds.dimensions[axis])
    # Confirm scalar initialization coordinates against the filename.
    for name in ("forecast_reference_time", "forecast_initial_time", "time"):
        if name in ds.variables and ds.variables[name].size == 1:
            absolute = datetime_values(ds.variables[name])
            if absolute is not None and abs(int(absolute.reshape(-1)[0]) - ns(init)) > 1000000000:
                raise ValueError(f"Filename initialization differs from {name}: {init}")
    override = ds.variables[args.step_var] if args.step_var else None
    if override is not None and coordinate_vector(override, axis) is None:
        raise ValueError(f"Lead coordinate {args.step_var!r} does not index {axis!r}.")
    coordinate_names = (
        [args.step_var] if override is not None else dict.fromkeys(["valid_time", axis, "time"])
    )
    for name in coordinate_names:
        if name not in ds.variables:
            continue
        var = ds.variables[name]
        selection = coordinate_vector(var, axis)
        if selection is None:
            continue
        absolute = datetime_values(var)
        if absolute is not None:
            absolute = np.asarray(absolute[selection]).reshape(-1)
            leads = (absolute.astype(np.longdouble) - ns(init)) / HOUR_NS
            if len(leads) == size:
                return validate_leads(axis, leads, args), "CF absolute valid times"
    var = override
    if var is None:
        var = next(
            (
                ds.variables[name]
                for name in (axis, "step", "forecast_hour", "forecast_period", "lead_time")
                if name in ds.variables and coordinate_vector(ds.variables[name], axis) is not None
            ),
            None,
        )
    if var is None:
        raise ValueError("No forecast lead coordinate found; specify --step-var.")
    raw = np.asarray(var[coordinate_vector(var, axis)], dtype=np.float64).reshape(-1)
    unit = args.step_unit
    saved_unit = plain(getattr(var, "units", "")).lower().strip()
    if unit == "auto":
        units = {
            "hours": "hours",
            "hour": "hours",
            "h": "hours",
            "seconds": "seconds",
            "second": "seconds",
            "s": "seconds",
            "nanoseconds": "nanoseconds",
            "ns": "nanoseconds",
        }
        unit = units.get(saved_unit)
        if unit is None:
            raise ValueError(
                f"Unrecognized lead units for {var.name}: {saved_unit!r}; specify --step-unit hours or index6h."
            )
    factors = {"hours": 1.0, "seconds": 1 / 3600.0, "nanoseconds": 1 / HOUR_NS, "index6h": 6.0}
    return validate_leads(axis, raw * factors[unit], args), f"{var.name}: {unit}"


def validate_leads(axis, leads, args):
    if not np.isfinite(leads).all():
        raise ValueError("Forecast leads must be finite.")
    rounded = np.rint(leads).astype("i8")
    if np.max(np.abs(leads - rounded)) > 1 / 3600:
        raise ValueError("Forecast leads must be whole hours.")
    if len(np.unique(rounded)) != len(rounded):
        raise ValueError("Duplicate forecast leads.")
    mapping = {int(value): i for i, value in enumerate(rounded)}
    required = LEADS if args.t0_source == "forecast" else LEADS[1:]
    missing = [int(value) for value in required if int(value) not in mapping]
    if missing:
        raise ValueError("Missing forecast leads (hours): " + str(missing))
    return axis, mapping


def grid_indices(source, target, cyclic=False):
    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 1 or not np.isfinite(source).all():
        raise ValueError("Wind inputs require finite one-dimensional coordinates.")
    distance = np.abs(source[None] - target[:, None])
    if cyclic:
        distance = np.abs((source[None] - target[:, None] + 180.0) % 360.0 - 180.0)
    indices = distance.argmin(axis=1)
    if np.any(distance[np.arange(len(target)), indices] > 1e-5) or len(np.unique(indices)) != len(
        indices
    ):
        raise ValueError(
            "Wind coordinates must match the training grid; spatial interpolation is not performed."
        )
    return indices


class WindReader:
    def __init__(self, metadata, args):
        self.meta, self.args = metadata, args
        self.cache = OrderedDict()

    def components(self, ds):
        lat = variable(ds, self.args.lat_var, LAT_NAMES, "latitude")
        lon = variable(ds, self.args.lon_var, LON_NAMES, "longitude")
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError("Only one-dimensional latitude/longitude coordinates are supported.")
        lat_dim, lon_dim = lat.dimensions[0], lon.dimensions[0]
        indices = (
            grid_indices(lat[:], self.meta["lat"]),
            grid_indices(lon[:], self.meta["lon"], True),
        )
        u = variable(ds, self.args.u_var, U_NAMES, "u10")
        v = variable(ds, self.args.v_var, V_NAMES, "v10")
        return u, v, lat_dim, lon_dim, indices

    def field(self, ds, var, lat_dim, lon_dim, indices, axis=None, step=None, stamp=None):
        units = plain(getattr(var, "units", "")).lower().replace(" ", "").replace("**", "^")
        if units not in (
            "",
            "m/s",
            "ms-1",
            "ms^-1",
            "m.s-1",
            "msec-1",
            "metersecond-1",
            "metressecond-1",
        ):
            raise ValueError(f"Expected wind units m/s for {var.name}, found {units!r}.")
        selections, remaining = [], []
        for dim in var.dimensions:
            size = len(ds.dimensions[dim])
            if dim == lat_dim:
                selections.append(indices[0])
                remaining.append(dim)
            elif dim == lon_dim:
                selections.append(indices[1])
                remaining.append(dim)
            elif dim == axis:
                selections.append(step)
            elif dim.lower() in MEMBER_DIMS:
                if dim not in ds.variables:
                    if size == 1:
                        selections.append(0)
                        continue
                    raise ValueError("Missing member coordinate for dimension: " + dim)
                ids = np.asarray(ds.variables[dim][:])
                match = np.flatnonzero(ids == self.args.member)
                if len(match) != 1:
                    raise ValueError(f"Cannot uniquely locate member {self.args.member}.")
                selections.append(int(match[0]))
            elif size == 1:
                selections.append(0)
            elif stamp is not None and dim in ds.variables:
                absolute = datetime_values(ds.variables[dim])
                match = (
                    np.flatnonzero(np.asarray(absolute).reshape(-1) == ns(stamp))
                    if absolute is not None
                    else []
                )
                if len(match) != 1:
                    raise ValueError(
                        f"Analysis time axis {dim} does not contain {iso(stamp)} exactly once."
                    )
                selections.append(int(match[0]))
            else:
                raise ValueError(f"Unsupported non-singleton dimension {dim} in {var.name}.")
        if set(remaining) != {lat_dim, lon_dim}:
            raise ValueError("Missing latitude/longitude dimensions in wind variable: " + var.name)
        array = np.ma.asarray(var[tuple(selections)]).astype("f4").filled(np.nan)
        if remaining == [lon_dim, lat_dim]:
            array = array.T
        if (
            array.shape != self.meta["ocean"].shape
            or not np.isfinite(array[self.meta["ocean"]]).all()
        ):
            raise ValueError(f"Missing ocean winds or incompatible shape in {var.name}.")
        return np.ascontiguousarray(array)

    def analysis(self, path, stamp):
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        with netCDF4.Dataset(str(path)) as ds:
            u, v, lat, lon, indices = self.components(ds)
            for name in ("valid_time", "time"):
                if name in ds.variables and ds.variables[name].size == 1:
                    value = datetime_values(ds.variables[name])
                    if value is not None and int(value.reshape(-1)[0]) != ns(stamp):
                        raise ValueError(f"Analysis time differs from filename: {path}")
            wind = np.stack([self.field(ds, var, lat, lon, indices, stamp=stamp) for var in (u, v)])
        self.cache[key] = wind
        if len(self.cache) > 48:
            self.cache.popitem(last=False)
        return wind

    def forecast(self, path, init):
        with netCDF4.Dataset(str(path)) as ds:
            u, v, lat, lon, indices = self.components(ds)
            (axis, mapping), interpretation = forecast_steps(ds, u, lat, lon, init, self.args)
            shape = (len(LEADS), 2, *self.meta["ocean"].shape)
            result = np.empty(shape, dtype="f4")
            for row, lead in enumerate(LEADS):
                if lead == 0 and self.args.t0_source == "analysis":
                    result[row] = np.nan
                    continue
                result[row] = np.stack(
                    [
                        self.field(ds, var, lat, lon, indices, axis=axis, step=mapping[int(lead)])
                        for var in (u, v)
                    ]
                )
        return result, interpretation
