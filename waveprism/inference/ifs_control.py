from __future__ import annotations

import argparse
import csv
import json
import os
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
    FORECAST_RE,
    HOUR_NS,
    LEADS,
    WindReader,
    date,
    index_files,
    iso,
    ns,
)
from ..inference.common import (
    NC_FIELDS,
    TIME_CALENDAR,
    TIME_UNITS,
    load_checkpoint,
    output_specs,
    predict_batch,
    restore_models,
    sha256_file,
    time_text,
)
from ..models.expert_forecaster import autocast_context


def create_daily_output(path, init, forcing, stage1, meta, args, expert_ids, signature, specs):
    arch = stage1["architecture"]
    h, w = meta["ocean"].shape
    with netCDF4.Dataset(str(path), "w", format="NETCDF4", clobber=False) as out:
        out.setncatts(
            dict(
                Conventions="CF-1.8",
                title="WavePrism operational 10-day wave forecast",
                run_signature=signature,
                forecast_initialization=iso(init),
                forecast_complete=np.int8(0),
                history_hours=int(arch["history_hours"]),
                t0_wind_source=args.t0_source,
                temporal_interpolation="linear in u10 and v10 from 6-hour anchors to hourly input",
                source_step_interpretation=forcing["interpretation"],
                forecast_wind_file=str(forcing["forecast_file"]),
                analysis_wind_files=json.dumps([str(p) for p in forcing["analysis_files"]]),
                future_analysis_used=np.int8(0),
                spatial_alignment="exact coordinate selection; no spatial interpolation",
            )
        )
        for dim, size in [
            ("step", len(LEADS)),
            ("lat", h),
            ("lon", w),
            ("all_expert", arch["num_experts"]),
        ]:
            out.createDimension(dim, size)
        if 1 in args.stages:
            out.createDimension("expert", len(expert_ids))
        if args.save_z:
            out.createDimension("latent_channel", arch["latent_channels"])

        def var(name, dtype, dims, data, **attrs):
            value = out.createVariable(name, dtype, dims, fill_value=False)
            if dims:
                value[:] = data
            else:
                value.assignValue(data)
            value.setncatts(attrs)
            return value

        var(
            "lat",
            "f8",
            ("lat",),
            meta["lat"],
            standard_name="latitude",
            units="degrees_north",
            axis="Y",
        )
        var(
            "lon",
            "f8",
            ("lon",),
            meta["lon"],
            standard_name="longitude",
            units="degrees_east",
            axis="X",
        )
        var(
            "forecast_reference_time",
            "f8",
            (),
            ns(init) / 1e9,
            standard_name="forecast_reference_time",
            units=TIME_UNITS,
            calendar=TIME_CALENDAR,
        )
        var(
            "step",
            "i4",
            ("step",),
            LEADS,
            standard_name="forecast_period",
            units="hours",
            long_name="Forecast lead time",
        )
        valid = ns(init) + LEADS.astype("i8") * HOUR_NS
        var(
            "valid_time",
            "f8",
            ("step",),
            valid.astype("f8") / 1e9,
            standard_name="time",
            units=TIME_UNITS,
            calendar=TIME_CALENDAR,
        )
        var("valid_time_utc", str, ("step",), np.asarray(time_text(valid), dtype=object))
        var("completed", "i1", ("step",), np.zeros(len(LEADS), dtype="i1"))
        var(
            "ocean_mask",
            "i1",
            ("lat", "lon"),
            meta["ocean"].astype("i1"),
            flag_values=np.asarray([0, 1], dtype="i1"),
            flag_meanings="land ocean",
        )
        var("all_expert", "i4", ("all_expert",), np.arange(1, arch["num_experts"] + 1))
        var(
            "all_expert_name",
            str,
            ("all_expert",),
            np.asarray(stage1["expert_names"], dtype=object),
        )
        if 1 in args.stages:
            var("expert", "i4", ("expert",), np.asarray(expert_ids, dtype="i4") + 1)
            var(
                "expert_name",
                str,
                ("expert",),
                np.asarray([stage1["expert_names"][i] for i in expert_ids], dtype=object),
            )
        if args.save_z:
            var("latent_channel", "i4", ("latent_channel",), np.arange(arch["latent_channels"]))
        for key, (shape, dtype, units) in specs.items():
            name, dimensions, description = NC_FIELDS[key]
            dims = tuple("step" if d == "time" else d for d in dimensions)
            chunks = (1, 1, h, w) if len(shape) == 4 else (1, h, w)
            value = out.createVariable(
                name,
                dtype,
                dims,
                fill_value=-1 if dtype == "i4" else np.nan,
                chunksizes=chunks,
                zlib=args.compression_level > 0,
                complevel=args.compression_level or 1,
                shuffle=True,
            )
            value.setncatts(
                dict(
                    units=units,
                    long_name=description,
                    coordinates="forecast_reference_time valid_time",
                )
            )
            if key in ("stage1/hs_experts", "stage2/hs_selected", "stage3/mu"):
                value.standard_name = "sea_surface_wave_significant_height"
        if args.save_wind:
            for i, name in enumerate(("u10", "v10")):
                value = out.createVariable(
                    name,
                    "f4",
                    ("step", "lat", "lon"),
                    fill_value=np.nan,
                    chunksizes=(1, h, w),
                    zlib=args.compression_level > 0,
                    complevel=args.compression_level or 1,
                )
                value[:] = forcing["future"][:, i]
                value.setncatts(
                    dict(
                        units="m s-1",
                        long_name="Effective 6-hour " + name + " forcing anchor",
                        coordinates="forecast_reference_time valid_time",
                    )
                )
        out.sync()


def run_day(init, forcing, model, stage1, meta, args, expert_ids, hashes):
    dest = Path(args.output_dir) / f"waveprism_{init:%Y%m%d}_00.nc"
    partial = Path(str(dest) + ".partial")
    signature = input_signature(init, forcing, args, hashes, meta, expert_ids)
    specs = output_specs(
        args, stage1["architecture"], len(LEADS), *meta["ocean"].shape, len(expert_ids)
    )

    def validate(out):
        out.set_auto_mask(False)
        if getattr(out, "run_signature", None) != signature:
            raise ValueError(
                "Existing output differs from the requested inputs/checkpoints: " + str(dest)
            )
        if not np.array_equal(out.variables["step"][:], LEADS):
            raise ValueError("Existing forecast leads differ.")
        for key, spec in specs.items():
            name = NC_FIELDS[key][0]
            if name not in out.variables or out.variables[name].shape != spec[0]:
                raise ValueError("Existing variable differs: " + name)

    if dest.exists():
        with netCDF4.Dataset(str(dest)) as out:
            validate(out)
            if getattr(out, "forecast_complete", 0) == 1 and np.all(
                out.variables["completed"][:] == 1
            ):
                return "skipped", str(dest)
        raise ValueError(
            "Output exists but is incomplete; inspect the corresponding .partial file."
        )
    if partial.exists() and not args.resume:
        raise FileExistsError("Incomplete output found; add --resume: " + str(partial))
    if not partial.exists():
        create_daily_output(
            partial, init, forcing, stage1, meta, args, expert_ids, signature, specs
        )
    if model is None:
        model = restore_models(args, stage1, meta, args.device_obj, hashes)
    windows = HourlyWindows(forcing, stage1)
    with netCDF4.Dataset(str(partial), "r+") as out:
        validate(out)
        pending = np.flatnonzero(out.variables["completed"][:] != 1)
        with torch.inference_mode():
            for start in tqdm(
                range(0, len(pending), args.batch_size),
                desc=f"Forecast {init:%Y-%m-%d}",
                unit="batch",
            ):
                rows = pending[start : start + args.batch_size]
                shape = (len(rows), windows.history, 2, *meta["ocean"].shape)
                batch = np.empty(shape, dtype="f4")
                for i, row in enumerate(rows):
                    batch[i] = windows.window(int(LEADS[row]))
                wind = torch.from_numpy(batch).to(args.device_obj)
                with autocast_context(args.device_obj.type, args.amp):
                    predictions = predict_batch(model, wind, meta, args, expert_ids)
                for key, values in predictions.items():
                    out.variables[NC_FIELDS[key][0]][rows, ...] = values
                out.sync()
                out.variables["completed"][rows] = np.int8(1)
                out.sync()
        out.forecast_complete = np.int8(1)
        out.sync()
    if dest.exists():
        raise FileExistsError(
            "Destination appeared during inference; retaining partial file: " + str(dest)
        )
    os.replace(partial, dest)
    return model, str(dest)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--analysis-dir", required=True)
    p.add_argument("--forecast-dir", required=True)
    p.add_argument("--static-npz", required=True)
    p.add_argument("--stage1-checkpoint", required=True)
    p.add_argument("--stage2-checkpoint", "--selector-checkpoint", default="")
    p.add_argument("--stage3-checkpoint", "--probability-checkpoint", default="")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--latest", action="store_true")
    mode.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument("--output-dir", default="./wave_forecast")
    p.add_argument("--stages", type=int, nargs="+", choices=(1, 2, 3), default=[3])
    p.add_argument("--expert-ids", type=int, nargs="+")
    p.add_argument("--save-z", action="store_true")
    p.add_argument("--save-selection-probabilities", action="store_true")
    p.add_argument("--save-wind", action="store_true")
    p.add_argument("--t0-source", choices=("forecast", "analysis"), default="forecast")
    p.add_argument("--u-var", default="")
    p.add_argument("--v-var", default="")
    p.add_argument("--lat-var", default="")
    p.add_argument("--lon-var", default="")
    p.add_argument("--step-var", default="")
    p.add_argument(
        "--step-unit",
        choices=("auto", "hours", "seconds", "nanoseconds", "index6h"),
        default="auto",
    )
    p.add_argument("--member", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compression-level", type=int, choices=range(10), default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args(argv)
    args.stages = sorted(set(args.stages))
    if args.batch_size < 1:
        p.error("batch-size must be positive.")
    if args.end_date and not args.start_date:
        p.error("--end-date requires --start-date.")
    if max(args.stages) == 3 and not args.stage3_checkpoint:
        p.error("Stage 3 requires --stage3-checkpoint.")
    if max(args.stages) == 2 and not args.stage2_checkpoint:
        p.error("Stage 2 requires --stage2-checkpoint.")
    if args.save_z and 1 not in args.stages:
        p.error("--save-z requires stage 1 in --stages.")
    if args.save_selection_probabilities and 2 not in args.stages:
        p.error("Selection probabilities require stage 2 in --stages.")
    if args.expert_ids and (
        1 not in args.stages or len(set(args.expert_ids)) != len(args.expert_ids)
    ):
        p.error("--expert-ids requires stage 1 and unique expert IDs.")
    args.latest = args.latest or not args.start_date
    args.dtype = "float32"
    args.data_h5 = ""
    args.save_observations = False
    args.device_obj = torch.device(
        ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if args.amp and args.device_obj.type != "cuda":
        p.error("--amp requires CUDA.")
    return args


def main(argv=None):
    args = parse_args(argv)
    stage1 = load_checkpoint(args.stage1_checkpoint)
    required = {"model", "architecture", "mean", "std", "ocean_mask", "expert_names"}
    if not required.issubset(stage1) or stage1["architecture"].get("version") != 2:
        raise ValueError("Expected a compatible Stage-1 v2 training checkpoint.")
    if (
        np.asarray(stage1["mean"]).shape != (2,)
        or np.asarray(stage1["std"]).shape != (2,)
        or not np.isfinite(stage1["mean"]).all()
        or not np.isfinite(stage1["std"]).all()
        or np.any(np.asarray(stage1["std"]) <= 0)
    ):
        raise ValueError("Invalid Stage-1 wind normalization statistics.")
    meta = geometry_from_static(stage1, args.static_npz)
    k = stage1["architecture"]["num_experts"]
    if len(stage1["expert_names"]) != k:
        raise ValueError("Expert count differs from the number of expert names.")
    expert_ids = [i - 1 for i in args.expert_ids] if args.expert_ids else list(range(k))
    if any(i < 0 or i >= k for i in expert_ids):
        raise ValueError(f"Expert IDs must lie between 1 and {k}.")
    analysis = index_files(args.analysis_dir, ANALYSIS_RE)
    forecast = index_files(args.forecast_dir, FORECAST_RE, True)
    if not forecast:
        raise FileNotFoundError("No matching 00 UTC forecast wind files found.")
    reader = WindReader(meta, args)
    prepared = {}
    notes = []
    if args.latest:
        dates = []
        for init in sorted(forecast, reverse=True):
            try:
                prepared[init] = prepare_forcing(
                    init,
                    forecast[init],
                    analysis,
                    reader,
                    stage1["architecture"]["history_hours"],
                    args,
                )
                dates = [init]
                break
            except (FileNotFoundError, ValueError, OSError) as exc:
                print(f"[LATEST SKIP] {init:%Y%m%d}: {exc}", flush=True)
                notes.append(
                    dict(init=iso(init), status="unavailable", output="", message=str(exc))
                )
        if not dates:
            raise RuntimeError(
                "No initialization has complete analysis history and a 240-hour forecast."
            )
    else:
        start = date(args.start_date)
        end = date(args.end_date or args.start_date)
        if end < start:
            raise ValueError("End date must not precede start date.")
        dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output)
    report = output / (
        "run_summary_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".csv"
    )
    hashes = {"stage1": sha256_file(args.stage1_checkpoint), "static": sha256_file(args.static_npz)}
    if max(args.stages) == 3:
        hashes["stage3"] = sha256_file(args.stage3_checkpoint)
        if args.stage2_checkpoint:
            hashes["stage2"] = sha256_file(args.stage2_checkpoint)
    elif max(args.stages) == 2:
        hashes["stage2"] = sha256_file(args.stage2_checkpoint)
    model = None
    results = list(notes)

    def save_report():
        with open(report, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["init", "status", "output", "message"])
            writer.writeheader()
            writer.writerows(results)

    print(
        f"Device={args.device_obj}; history={stage1['architecture']['history_hours']}h; output leads=0:6:240 (41)",
        flush=True,
    )
    for init in dates:
        try:
            if init not in forecast:
                raise FileNotFoundError(
                    f"Missing forecast file: ifs_control_u10v10_{init:%Y%m%d}_00.nc (optional _0p5deg suffix)."
                )
            forcing = (
                prepared.pop(init)
                if init in prepared
                else prepare_forcing(
                    init,
                    forecast[init],
                    analysis,
                    reader,
                    stage1["architecture"]["history_hours"],
                    args,
                )
            )
            print(
                f"[INIT] {iso(init)}; analysis files={len(forcing['analysis_files'])}; step={forcing['interpretation']}",
                flush=True,
            )
            if args.check_only:
                results.append(
                    dict(
                        init=iso(init),
                        status="checked",
                        output="",
                        message="Input forcing complete; no model inference",
                    )
                )
            else:
                outcome, path = run_day(
                    init, forcing, model, stage1, meta, args, expert_ids, hashes
                )
                status = "skipped" if isinstance(outcome, str) else "success"
                if status == "success":
                    model = outcome
                results.append(
                    dict(
                        init=iso(init),
                        status=status,
                        output=path,
                        message="41 leads: 0,6,...,240 h",
                    )
                )
                print(f"[{status.upper()}] {path}", flush=True)
            del forcing
        except Exception as exc:
            results.append(
                dict(
                    init=iso(init),
                    status="error",
                    output="",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            print(f"[ERROR] {init:%Y%m%d}: {type(exc).__name__}: {exc}", flush=True)
        save_report()
    print("Run summary:", report, flush=True)
    return 1 if any(row["status"] == "error" for row in results) else 0


if __name__ == "__main__":
    sys.exit(main())
