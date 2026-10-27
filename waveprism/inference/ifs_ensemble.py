from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4
import numpy as np
import torch
from tqdm.auto import tqdm

from ..data.forcing import HourlyWindows, geometry_from_static, input_signature, prepare_forcing
from ..data.ifs_wind import (
    ANALYSIS_RE,
    HOUR_NS,
    LAT_NAMES,
    LEADS,
    LON_NAMES,
    MEMBER_DIMS,
    U_NAMES,
    WindReader,
    date,
    index_files,
    iso,
    ns,
    variable,
)
from ..inference.common import (
    TIME_CALENDAR,
    TIME_UNITS,
    load_checkpoint,
    predict_batch,
    restore_models,
    sha256_file,
)
from ..models.expert_forecaster import autocast_context

MEMBER_DIM_NAMES = ("member", "number", "members", "ensemble", "realization")

ENSEMBLE_RE = re.compile(r"^ifs_ens_(?:u10v10|uv10|u10_v10)_(\d{8})_(\d{2})(?:_0p5deg)?\.nc$", re.I)


def parse_date(value: str) -> datetime:
    return date(value)


def index_ensemble_files(directory: str) -> dict[datetime, Path]:
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(root)
    files: dict[datetime, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        match = ENSEMBLE_RE.fullmatch(path.name)
        if not match:
            continue
        stamp = datetime.strptime("".join(match.groups()), "%Y%m%d%H")
        # Keep the same 00 UTC initialization convention as the existing
        # operational workflow. This can be relaxed if 12 UTC is later needed.
        if stamp.hour != 0:
            continue
        if stamp in files:
            raise ValueError(
                f"Multiple IFS ensemble files for the same 00 UTC initialization: {files[stamp]} / {path}"
            )
        files[stamp] = path.resolve()
    return files


def get_member_ids(path: Path, args) -> tuple[list[int], str]:
    """Read member IDs from the member coordinate of the IFS ENS NetCDF."""
    with netCDF4.Dataset(str(path), "r") as ds:
        u = variable(ds, args.u_var, U_NAMES, "u10")
        lat = variable(ds, args.lat_var, LAT_NAMES, "latitude")
        lon = variable(ds, args.lon_var, LON_NAMES, "longitude")
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError("IFS ensemble winds require one-dimensional coordinates.")
        lat_dim, lon_dim = lat.dimensions[0], lon.dimensions[0]
        member_dims = [
            d for d in u.dimensions if d.lower() in MEMBER_DIMS and d not in (lat_dim, lon_dim)
        ]
        if len(member_dims) != 1:
            raise ValueError(f"Cannot uniquely identify IFS member dimension: {u.dimensions}")
        member_dim = member_dims[0]
        if member_dim not in ds.variables:
            raise ValueError(f"Missing member coordinate for dimension {member_dim!r}.")
        raw = np.asarray(ds.variables[member_dim][:]).reshape(-1)
        numeric = raw.astype(np.float64)
        if not np.isfinite(numeric).all() or not np.allclose(numeric, np.rint(numeric)):
            raise ValueError(f"Member IDs must be finite integers: {raw!r}")
        ids = [int(x) for x in np.rint(numeric)]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IFS member IDs: {ids}")
        if args.control_member not in ids:
            raise ValueError(f"IFS file has no control member {args.control_member}; found {ids}.")
        return ids, member_dim


def build_args(cli) -> argparse.Namespace:
    """Create the Namespace expected by the existing inference functions."""
    device = torch.device(
        ("cuda:0" if torch.cuda.is_available() else "cpu") if cli.device == "auto" else cli.device
    )
    if cli.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA.")
    return argparse.Namespace(
        analysis_dir=cli.analysis_dir,
        forecast_dir=cli.forecast_dir,
        static_npz=cli.static_npz,
        stage1_checkpoint=cli.stage1_checkpoint,
        stage2_checkpoint=cli.stage2_checkpoint,
        stage3_checkpoint=cli.stage3_checkpoint,
        t0_source=cli.t0_source,
        u_var=cli.u_var,
        v_var=cli.v_var,
        lat_var=cli.lat_var,
        lon_var=cli.lon_var,
        step_var=cli.step_var,
        step_unit=cli.step_unit,
        member=cli.control_member,
        control_member=cli.control_member,
        batch_size=cli.batch_size,
        device_obj=device,
        amp=cli.amp,
        stages=[3],
        expert_ids=None,
        save_z=False,
        save_selection_probabilities=False,
        save_wind=False,
        save_observations=False,
        compression_level=cli.compression_level,
        dtype="float32",
        data_h5="",
        resume=cli.resume,
        check_only=cli.check_only,
        output_dir=cli.output_dir,
    )


def create_output_file(
    path: Path,
    init: datetime,
    forcing: dict,
    stage1: dict,
    meta: dict,
    args: argparse.Namespace,
    member_ids: list[int],
    signature: str,
) -> None:
    arch = stage1["architecture"]
    nstep = len(LEADS)
    nmember = len(member_ids)
    h, w = meta["ocean"].shape
    lat_chunk, lon_chunk = min(h, 128), min(w, 180)
    comp = int(args.compression_level)
    common_var_kwargs = dict(zlib=comp > 0, complevel=comp if comp > 0 else 1, shuffle=True)

    with netCDF4.Dataset(str(path), "w", format="NETCDF4", clobber=False) as out:
        out.setncatts(
            dict(
                Conventions="CF-1.8",
                title="WavePrism probabilistic operational forecast driven by IFS ENS winds",
                format_version="2.0",
                run_signature=signature,
                forecast_initialization=iso(init),
                forecast_complete=np.int8(0),
                history_hours=int(arch["history_hours"]),
                t0_wind_source=args.t0_source,
                temporal_interpolation="linear in u10 and v10 from 6-hour anchors to hourly input",
                forecast_wind_file=str(forcing["forecast_file"]),
                analysis_wind_files=json.dumps([str(p) for p in forcing["analysis_files"]]),
                future_analysis_used=np.int8(0),
                spatial_alignment="exact coordinate selection; no spatial interpolation",
                control_member=int(args.control_member),
                ensemble_members=json.dumps(member_ids),
                ensemble_weights="equal weights across selected IFS members",
                mixture_definition=(
                    "Equal-weight mixture of member-conditional Gaussian predictions"
                ),
                sigma_definition=(
                    "sqrt(mean_m(sigma_m^2) + mean_m((mu_m - mean(mu_m))^2)); "
                    "standard deviation of the Gaussian mixture"
                ),
                summary_pair_note=(
                    "stage3_mu is the control-member mean while stage3_sigma is the "
                    "full-mixture standard deviation. This pair is a requested "
                    "summary and is not an exact single-Gaussian representation."
                ),
            )
        )
        for dim, size in (("step", nstep), ("member", nmember), ("lat", h), ("lon", w)):
            out.createDimension(dim, int(size))

        def create(name, dtype, dims, data=None, chunks=None, fill_value=None, **attrs):
            # NetCDF scalar variables cannot use chunked compression.
            kwargs = dict(common_var_kwargs) if len(dims) > 0 else {}
            if chunks is not None:
                kwargs["chunksizes"] = chunks
            if fill_value is not None:
                kwargs["fill_value"] = fill_value
            var = out.createVariable(name, dtype, dims, **kwargs)
            if data is not None:
                if len(dims) == 0:
                    var.assignValue(np.asarray(data).item())
                else:
                    var[:] = data
            if attrs:
                var.setncatts(attrs)
            return var

        create(
            "lat",
            "f8",
            ("lat",),
            meta["lat"],
            standard_name="latitude",
            units="degrees_north",
            axis="Y",
        )
        create(
            "lon",
            "f8",
            ("lon",),
            meta["lon"],
            standard_name="longitude",
            units="degrees_east",
            axis="X",
        )
        create(
            "member",
            "i4",
            ("member",),
            np.asarray(member_ids, dtype=np.int32),
            long_name="IFS ensemble member identifier",
            comment="0 denotes the control member in the project's IFS ENS files.",
        )
        create(
            "forecast_reference_time",
            "f8",
            (),
            np.asarray(ns(init) / 1e9),
            standard_name="forecast_reference_time",
            units=TIME_UNITS,
            calendar=TIME_CALENDAR,
        )
        create(
            "step",
            "i4",
            ("step",),
            LEADS,
            standard_name="forecast_period",
            units="hours",
            long_name="Forecast lead time",
        )
        valid_ns = ns(init) + LEADS.astype("i8") * HOUR_NS
        create(
            "valid_time",
            "f8",
            ("step",),
            valid_ns.astype("f8") / 1e9,
            standard_name="time",
            units=TIME_UNITS,
            calendar=TIME_CALENDAR,
        )
        create(
            "ocean_mask",
            "i1",
            ("lat", "lon"),
            meta["ocean"].astype("i1"),
            flag_values=np.asarray([0, 1], dtype="i1"),
            flag_meanings="land ocean",
        )
        done = out.createVariable("completed", "i1", ("member", "step"))
        done[:] = np.zeros((nmember, nstep), dtype=np.int8)
        done.long_name = "Member and forecast lead inference completion flag"
        done.flag_values = np.asarray([0, 1], dtype="i1")
        done.flag_meanings = "incomplete complete"

        chunks4 = (1, 1, lat_chunk, lon_chunk)
        chunks3 = (1, lat_chunk, lon_chunk)
        create(
            "stage3_mu_member",
            "f4",
            ("step", "member", "lat", "lon"),
            chunks=chunks4,
            fill_value=np.float32(np.nan),
            units="m",
            long_name="Stage-3 predictive mean for each IFS wind member",
            standard_name="sea_surface_wave_significant_height",
        )
        create(
            "stage3_sigma_member",
            "f4",
            ("step", "member", "lat", "lon"),
            chunks=chunks4,
            fill_value=np.float32(np.nan),
            units="m",
            long_name="Stage-3 conditional predictive standard deviation for each IFS wind member",
        )

        create(
            "stage3_mu",
            "f4",
            ("step", "lat", "lon"),
            chunks=chunks3,
            fill_value=np.float32(np.nan),
            units="m",
            long_name="Predictive mean from IFS control-member-driven WavePrism",
            standard_name="sea_surface_wave_significant_height",
            comment=f"Uses IFS member {args.control_member}; this is the requested deterministic center.",
        )
        create(
            "stage3_sigma",
            "f4",
            ("step", "lat", "lon"),
            chunks=chunks3,
            fill_value=np.float32(np.nan),
            units="m",
            long_name="Standard deviation of the equally weighted Gaussian mixture",
        )
        out.sync()


def validate_partial(path: Path, signature: str, member_ids: list[int]) -> None:
    with netCDF4.Dataset(str(path), "r") as out:
        if getattr(out, "run_signature", None) != signature:
            raise ValueError(
                "Partial output differs from the current inputs, checkpoints, or member selection."
            )
        if not np.array_equal(out.variables["step"][:], LEADS):
            raise ValueError("Partial output forecast leads differ.")
        if not np.array_equal(out.variables["member"][:], np.asarray(member_ids, dtype=np.int32)):
            raise ValueError("Partial output IFS member IDs differ.")
        required = (
            "completed",
            "stage3_mu_member",
            "stage3_sigma_member",
            "stage3_mu",
            "stage3_sigma",
        )
        for name in required:
            if name not in out.variables:
                raise ValueError(f"Partial output is missing variable: {name}")


def mixture_summary(mu_members, sigma_members, control_index):
    """Return the control mean and full-mixture standard deviation."""
    mu = np.asarray(mu_members, dtype=np.float64)
    sigma = np.asarray(sigma_members, dtype=np.float64)
    valid = np.isfinite(mu).all(axis=0) & np.isfinite(sigma).all(axis=0) & (sigma > 0).all(axis=0)
    mixture_mean = mu.mean(axis=0)
    variance = np.mean(sigma**2, axis=0) + np.mean((mu - mixture_mean) ** 2, axis=0)
    return {
        "stage3_mu": np.where(valid, mu[control_index], np.nan).astype("f4"),
        "stage3_sigma": np.where(valid, np.sqrt(variance), np.nan).astype("f4"),
    }


def aggregate_and_finalize(partial, dest, signature, member_ids, control_member):
    """Aggregate resumable member results and export only the two final wave fields."""
    with netCDF4.Dataset(str(partial), "r+") as out:
        if out.run_signature != signature:
            raise ValueError("Run signature differs before aggregation.")
        if not np.all(out["completed"][:] == 1):
            raise RuntimeError("Member predictions remain incomplete; retaining partial output.")
        control_index = member_ids.index(control_member)
        for row in tqdm(range(len(LEADS)), desc="Aggregate mixture", unit="lead"):
            summary = mixture_summary(
                out["stage3_mu_member"][row].filled(np.nan),
                out["stage3_sigma_member"][row].filled(np.nan),
                control_index,
            )
            for name, values in summary.items():
                out[name][row] = values
        out.forecast_complete = np.int8(1)
        out.sync()

    finalizing = Path(str(dest) + ".finalizing")
    with netCDF4.Dataset(str(partial)) as source, netCDF4.Dataset(str(finalizing), "w") as target:
        target.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
        for name, dim in source.dimensions.items():
            target.createDimension(name, len(dim))
        for name, var in source.variables.items():
            if name in ("stage3_mu_member", "stage3_sigma_member"):
                continue
            options = dict(fill_value=getattr(var, "_FillValue", False))
            if var.ndim and var.dtype != str:
                filters = var.filters()
                options.update(zlib=filters["zlib"], complevel=filters["complevel"] or 1)
            result = target.createVariable(name, var.datatype, var.dimensions, **options)
            result.setncatts(
                {attr: var.getncattr(attr) for attr in var.ncattrs() if attr != "_FillValue"}
            )
            if var.ndim >= 3:
                for row in range(var.shape[0]):
                    result[row] = var[row]
            else:
                result[...] = var[...]
    if dest.exists():
        raise FileExistsError(f"Destination already exists; retaining partial output: {dest}")
    os.replace(finalizing, dest)
    partial.unlink()
    return str(dest)


def run_day(
    init: datetime,
    forecast_path: Path,
    control_forcing: dict,
    analysis_index: dict,
    reader,
    model,
    stage1: dict,
    meta: dict,
    args: argparse.Namespace,
    member_ids: list[int],
    expert_ids: list[int],
    hashes: dict[str, str],
) -> tuple[object, str]:
    dest = Path(args.output_dir) / f"waveprism_ensemble_{init:%Y%m%d}_00.nc"
    partial = Path(str(dest) + ".partial")

    # Base signature records the exact forecast/analysis inputs and checkpoints.
    args.member = args.control_member
    base_signature = json.loads(
        input_signature(init, control_forcing, args, hashes, meta, expert_ids)
    )
    signature = json.dumps(
        {
            "base": base_signature,
            "ensemble_member_ids": member_ids,
            "control_member": args.control_member,
            "aggregation": "equal-weight Gaussian mixture; law of total variance",
            "output_version": 2,
        },
        sort_keys=True,
    )

    if dest.exists():
        with netCDF4.Dataset(str(dest), "r") as out:
            if getattr(out, "run_signature", None) != signature:
                raise ValueError(
                    "Existing output differs from the requested inputs, checkpoints, or members: "
                    + str(dest)
                )
            if getattr(out, "forecast_complete", 0) == 1 and np.all(
                np.asarray(out.variables["completed"][:]) == 1
            ):
                return "skipped", str(dest)
        raise ValueError("Final output is incomplete; inspect the corresponding .partial file.")

    if partial.exists() and not args.resume:
        raise FileExistsError("Incomplete output found; add --resume: " + str(partial))
    if not partial.exists():
        create_output_file(
            partial, init, control_forcing, stage1, meta, args, member_ids, signature
        )
    else:
        validate_partial(partial, signature, member_ids)

    if model is None:
        model = restore_models(args, stage1, meta, args.device_obj, hashes)

    # Process one member at a time to avoid allocating a member x history x grid
    # tensor on the GPU. The analysis history is cached and shared across members.
    for member_pos, member_id in enumerate(member_ids):
        with netCDF4.Dataset(str(partial), "r") as out:
            pending = np.flatnonzero(np.asarray(out.variables["completed"][member_pos, :]) != 1)
        if len(pending) == 0:
            print(f"[MEMBER SKIP] IFS member={member_id}: all leads complete", flush=True)
            continue

        args.member = int(member_id)
        if member_id == args.control_member:
            forcing = control_forcing
        else:
            forcing = prepare_forcing(
                init,
                forecast_path,
                analysis_index,
                reader,
                stage1["architecture"]["history_hours"],
                args,
            )
        windows = HourlyWindows(forcing, stage1)
        print(f"[MEMBER] IFS member={member_id}; pending leads={len(pending)}", flush=True)

        with netCDF4.Dataset(str(partial), "r+") as out:
            with torch.inference_mode():
                for start in tqdm(
                    range(0, len(pending), args.batch_size),
                    desc=f"{init:%Y-%m-%d} member {member_id}",
                    unit="batch",
                ):
                    rows = pending[start : start + args.batch_size]
                    batch_shape = (len(rows), windows.history, 2, *meta["ocean"].shape)
                    wind_batch = np.empty(batch_shape, dtype=np.float32)
                    for i, row in enumerate(rows):
                        wind_batch[i] = windows.window(int(LEADS[row]))
                    wind = torch.from_numpy(wind_batch).to(args.device_obj)
                    with autocast_context(args.device_obj.type, args.amp):
                        predictions = predict_batch(model, wind, meta, args, expert_ids)
                    mu_values = np.asarray(predictions["stage3/mu"], dtype=np.float32)
                    sigma_values = np.asarray(predictions["stage3/sigma"], dtype=np.float32)
                    # Store the exact conditional Gaussian component for every member.
                    for j, row in enumerate(rows):
                        out.variables["stage3_mu_member"][int(row), member_pos, :, :] = mu_values[j]
                        out.variables["stage3_sigma_member"][int(row), member_pos, :, :] = (
                            sigma_values[j]
                        )
                    out.sync()
                    out.variables["completed"][member_pos, rows] = np.int8(1)
                    out.sync()
                    del wind, wind_batch, predictions, mu_values, sigma_values
        del windows
        if member_id != args.control_member:
            del forcing

    args.member = args.control_member
    final_path = aggregate_and_finalize(partial, dest, signature, member_ids, args.control_member)
    return model, final_path


def parse_cli():
    p = argparse.ArgumentParser(
        description="WavePrism operational probabilistic forecast using IFS ENS wind members",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--analysis-dir",
        required=True,
        help="Directory containing 10m_wind_YYYYMMDD_HHMM.nc; searched recursively.",
    )
    p.add_argument(
        "--forecast-dir",
        required=True,
        help="Directory containing ifs_ens_u10v10_YYYYMMDD_00.nc (optional _0p5deg suffix).",
    )
    p.add_argument("--static-npz", required=True)
    p.add_argument("--stage1-checkpoint", required=True)
    p.add_argument(
        "--stage2-checkpoint",
        default="",
        help="Optional selector checkpoint for verifying consistency with Stage 3.",
    )
    p.add_argument("--stage3-checkpoint", required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest 00 UTC initialization with complete analysis history.",
    )
    mode.add_argument("--start-date", help="First initialization date: YYYYMMDD or YYYY-MM-DD.")
    p.add_argument(
        "--end-date", help="Last initialization date (inclusive); requires --start-date."
    )
    p.add_argument("--output-dir", default="./waveprism_ensemble_prob")
    p.add_argument(
        "--member-ids",
        type=int,
        nargs="+",
        default=None,
        help="Member subset; defaults to all members and must include the control member.",
    )
    p.add_argument("--control-member", type=int, default=0)
    p.add_argument("--t0-source", choices=("forecast", "analysis"), default="forecast")
    p.add_argument("--u-var", default="")
    p.add_argument("--v-var", default="")
    p.add_argument("--lat-var", default="")
    p.add_argument("--lon-var", default="")
    p.add_argument(
        "--step-var",
        default="",
        help="Forecast lead coordinate override; otherwise detected automatically.",
    )
    p.add_argument(
        "--step-unit",
        choices=("auto", "hours", "seconds", "nanoseconds", "index6h"),
        default="auto",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Forecast leads per model batch; start with 1 on a global grid.",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compression-level", type=int, choices=range(10), default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--check-only", action="store_true", help="Validate inputs without model inference."
    )
    cli = p.parse_args()
    if cli.end_date and not cli.start_date:
        p.error("--end-date requires --start-date.")
    if cli.start_date and cli.end_date:
        if parse_date(cli.end_date) < parse_date(cli.start_date):
            p.error("End date must not precede start date.")
    if cli.batch_size < 1:
        p.error("--batch-size must be positive.")
    if cli.member_ids is not None and len(cli.member_ids) != len(set(cli.member_ids)):
        p.error("--member-ids must be unique.")
    return cli


def main() -> int:
    cli = parse_cli()
    args = build_args(cli)
    stage1 = load_checkpoint(args.stage1_checkpoint)
    required = {"model", "architecture", "mean", "std", "ocean_mask", "expert_names"}
    if not required.issubset(stage1) or stage1["architecture"].get("version") != 2:
        raise ValueError("Expected a compatible Stage-1 v2 checkpoint.")
    if (
        np.asarray(stage1["mean"]).shape != (2,)
        or np.asarray(stage1["std"]).shape != (2,)
        or not np.isfinite(stage1["mean"]).all()
        or not np.isfinite(stage1["std"]).all()
        or np.any(np.asarray(stage1["std"]) <= 0)
    ):
        raise ValueError("Invalid Stage-1 normalization statistics.")

    meta = geometry_from_static(stage1, args.static_npz)
    n_experts = int(stage1["architecture"]["num_experts"])
    if len(stage1["expert_names"]) != n_experts:
        raise ValueError("Stage-1 expert count differs from expert names.")
    expert_ids = list(range(n_experts))

    analysis_index = index_files(args.analysis_dir, ANALYSIS_RE)
    forecast_index = index_ensemble_files(args.forecast_dir)
    if not forecast_index:
        raise FileNotFoundError(
            "No IFS ensemble files found; expected ifs_ens_u10v10_YYYYMMDD_00.nc (optional _0p5deg suffix)."
        )

    if cli.latest or not cli.start_date:
        candidate_dates = sorted(forecast_index, reverse=True)
        dates = []
        prepared_control = {}
        reader = WindReader(meta, args)
        for init in candidate_dates:
            path = forecast_index[init]
            try:
                ids, _ = get_member_ids(path, args)
                selected_ids = list(cli.member_ids) if cli.member_ids is not None else ids
                if args.control_member not in selected_ids:
                    raise ValueError("--member-ids must include the control member.")
                missing_ids = [m for m in selected_ids if m not in ids]
                if missing_ids:
                    raise ValueError(f"Requested members are absent from the file: {missing_ids}")
                args.member = args.control_member
                prepared_control[init] = prepare_forcing(
                    init,
                    path,
                    analysis_index,
                    reader,
                    stage1["architecture"]["history_hours"],
                    args,
                )
                dates = [init]
                break
            except (FileNotFoundError, ValueError, OSError) as exc:
                print(f"[LATEST SKIP] {init:%Y%m%d}: {exc}", flush=True)
        if not dates:
            raise RuntimeError(
                "No initialization has both ensemble forecasts and complete analysis history."
            )
    else:
        start = parse_date(cli.start_date)
        end = parse_date(cli.end_date or cli.start_date)
        dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        prepared_control = {}
        reader = WindReader(meta, args)

    output_dir = Path(cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    report_path = output_dir / (
        "run_summary_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".csv"
    )

    hashes = {
        "stage1": sha256_file(args.stage1_checkpoint),
        "static": sha256_file(args.static_npz),
        "stage3": sha256_file(args.stage3_checkpoint),
    }
    if args.stage2_checkpoint:
        hashes["stage2"] = sha256_file(args.stage2_checkpoint)

    results = []
    model = None
    print(
        f"Device={args.device_obj}; history={stage1['architecture']['history_hours']} h; "
        f"leads=0:6:240 h ({len(LEADS)}); IFS ensemble mixture",
        flush=True,
    )
    print(f"Forecast files={len(forecast_index)}; output={output_dir}", flush=True)

    for init in dates:
        try:
            if init not in forecast_index:
                raise FileNotFoundError(
                    f"Missing IFS ensemble file: ifs_ens_u10v10_{init:%Y%m%d}_00.nc (optional _0p5deg suffix)."
                )
            forecast_path = forecast_index[init]
            all_ids, member_dim = get_member_ids(forecast_path, args)
            member_ids = list(cli.member_ids) if cli.member_ids is not None else all_ids
            if args.control_member not in member_ids:
                raise ValueError("Member selection must include the control member.")
            missing_ids = [m for m in member_ids if m not in all_ids]
            if missing_ids:
                raise ValueError(f"Requested members are absent from the IFS file: {missing_ids}")
            if len(member_ids) < 2:
                raise ValueError(
                    "Select at least two IFS members for ensemble mixture forecasting."
                )

            args.member = args.control_member
            control_forcing = prepared_control.pop(init, None)
            if control_forcing is None:
                control_forcing = prepare_forcing(
                    init,
                    forecast_path,
                    analysis_index,
                    reader,
                    stage1["architecture"]["history_hours"],
                    args,
                )
            print(
                f"[INIT] {iso(init)}; file={forecast_path.name}; member_dim={member_dim}; "
                f"members={member_ids}; history analysis files={len(control_forcing['analysis_files'])}",
                flush=True,
            )

            if cli.check_only:
                # Validate that each requested member can be read and has complete leads.
                for mid in member_ids:
                    args.member = int(mid)
                    _ = reader.forecast(forecast_path, init)
                args.member = args.control_member
                results.append(
                    {
                        "init": iso(init),
                        "status": "checked",
                        "output": "",
                        "message": f"{len(member_ids)} members; 41 leads; no inference",
                    }
                )
                print(f"[CHECKED] {init:%Y%m%d}: {len(member_ids)} members", flush=True)
            else:
                outcome, output_path = run_day(
                    init,
                    forecast_path,
                    control_forcing,
                    analysis_index,
                    reader,
                    model,
                    stage1,
                    meta,
                    args,
                    member_ids,
                    expert_ids,
                    hashes,
                )
                status = "skipped" if isinstance(outcome, str) else "success"
                if status == "success":
                    model = outcome
                results.append(
                    {
                        "init": iso(init),
                        "status": status,
                        "output": output_path,
                        "message": f"{len(member_ids)} members; 41 leads; mu=control, sigma=mixture std",
                    }
                )
                print(f"[{status.upper()}] {output_path}", flush=True)
            del control_forcing
        except Exception as exc:
            results.append(
                {
                    "init": iso(init),
                    "status": "error",
                    "output": "",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[ERROR] {init:%Y%m%d}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            args.member = args.control_member
            with open(report_path, "w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["init", "status", "output", "message"])
                writer.writeheader()
                writer.writerows(results)

    print(f"Run summary: {report_path}", flush=True)
    return 1 if any(row["status"] == "error" for row in results) else 0


if __name__ == "__main__":
    sys.exit(main())
