from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import netCDF4
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data.era5 import InferenceDataset, read_era5_metadata
from ..models.expert_forecaster import autocast_context
from .common import (
    FORMAT_VERSION,
    NC_FIELDS,
    TIME_CALENDAR,
    TIME_UNITS,
    array_hash,
    load_checkpoint,
    output_specs,
    predict_batch,
    restore_models,
    sha256_file,
    time_text,
)


def initialize_output(path, signature, specs, metadata, stage1, args, expert_ids):
    """Create a genuine NetCDF4 dataset with named dimensions and CF coordinates."""
    count = len(metadata["ids"])
    arch = stage1["architecture"]
    h, w = metadata["ocean"].shape
    target_times = metadata["times"][metadata["ids"]]
    with netCDF4.Dataset(str(path), "w", format="NETCDF4", clobber=False) as out:
        out.setncatts(
            dict(
                Conventions="CF-1.8",
                title="WavePrism staged global wave hindcast inference from hourly ERA5 winds",
                format_version=FORMAT_VERSION,
                run_signature=signature,
                settings_json=json.dumps(vars(args), sort_keys=True),
                history_hours=int(arch["history_hours"]),
                architecture_json=json.dumps(arch, sort_keys=True),
                time_selection=metadata["split"],
                inference_complete=np.int8(0),
                input_source="Daily ERA5 NetCDF u10/v10; no forecast wind fields; no HDF5 input dataset",
                input_file_manifest_sha256=metadata["source_manifest_hash"],
                requested_count=metadata["requested_count"],
                skipped_incomplete_history_count=len(metadata["skipped_ids"]),
                expert_index_convention="1-based Stage-1 decoder index; -1 = invalid/land",
                geospatial_lat_min=float(metadata["lat"].min()),
                geospatial_lat_max=float(metadata["lat"].max()),
                geospatial_lon_min=float(metadata["lon"].min()),
                geospatial_lon_max=float(metadata["lon"].max()),
                time_coverage_start=time_text(target_times[:1])[0],
                time_coverage_end=time_text(target_times[-1:])[0],
            )
        )
        for name, size in [
            ("time", count),
            ("lat", h),
            ("lon", w),
            ("all_expert", arch["num_experts"]),
            ("wind_component", arch["wind_channels"]),
            ("skipped_time", len(metadata["skipped_ids"])),
        ]:
            out.createDimension(name, int(size))
        if 1 in args.stages:
            out.createDimension("expert", len(expert_ids))
        if args.save_z:
            out.createDimension("latent_channel", int(arch["latent_channels"]))

        def variable(name, dtype, dims, data, attrs=None):
            var = out.createVariable(name, dtype, dims, fill_value=False)
            var[:] = data
            if attrs:
                var.setncatts(attrs)
            return var

        variable(
            "lat",
            "f8",
            ("lat",),
            metadata["lat"],
            dict(standard_name="latitude", long_name="Latitude", units="degrees_north", axis="Y"),
        )
        variable(
            "lon",
            "f8",
            ("lon",),
            metadata["lon"],
            dict(standard_name="longitude", long_name="Longitude", units="degrees_east", axis="X"),
        )
        variable(
            "time",
            "f8",
            ("time",),
            target_times.astype("f8") / 1e9,
            dict(
                standard_name="time",
                long_name="Prediction target time in UTC",
                units=TIME_UNITS,
                calendar=TIME_CALENDAR,
                axis="T",
            ),
        )
        variable(
            "time_utc",
            str,
            ("time",),
            np.asarray(time_text(target_times), dtype=object),
            dict(long_name="ISO 8601 prediction target time in UTC"),
        )
        variable(
            "source_time_ns",
            "i8",
            ("time",),
            target_times,
            dict(
                long_name="Exact source Unix timestamp in nanoseconds",
                comment="Original int64 timestamps; the time coordinate provides CF time decoding.",
            ),
        )
        variable(
            "source_index",
            "i8",
            ("time",),
            metadata["ids"],
            dict(long_name="Zero-based index in the chronologically sorted ERA5 hourly time axis"),
        )
        variable(
            "completed",
            "i1",
            ("time",),
            np.zeros(count, dtype="i1"),
            dict(
                long_name="Output commit status for each target time",
                flag_values=np.asarray([0, 1], dtype="i1"),
                flag_meanings="pending complete",
            ),
        )
        variable(
            "ocean_mask",
            "i1",
            ("lat", "lon"),
            metadata["ocean"].astype("i1"),
            dict(
                long_name="Frozen Stage-1 ocean mask",
                flag_values=np.asarray([0, 1], dtype="i1"),
                flag_meanings="land ocean",
            ),
        )
        variable(
            "all_expert",
            "i4",
            ("all_expert",),
            np.arange(1, arch["num_experts"] + 1),
            dict(long_name="All Stage-1 expert decoder indices, starting at one"),
        )
        variable(
            "all_expert_name",
            str,
            ("all_expert",),
            np.asarray(stage1["expert_names"], dtype=object),
        )
        if 1 in args.stages:
            variable(
                "expert",
                "i4",
                ("expert",),
                np.asarray(expert_ids, dtype="i4") + 1,
                dict(long_name="Exported Stage-1 expert decoder indices, starting at one"),
            )
            variable(
                "expert_name",
                str,
                ("expert",),
                np.asarray([stage1["expert_names"][i] for i in expert_ids], dtype=object),
            )
        if args.save_z:
            variable(
                "latent_channel",
                "i4",
                ("latent_channel",),
                np.arange(arch["latent_channels"]),
                dict(long_name="Zero-based encoder latent feature channel"),
            )
        variable(
            "wind_component",
            "i4",
            ("wind_component",),
            np.arange(arch["wind_channels"]),
            dict(long_name="Zero-based input wind component channel"),
        )
        variable(
            "input_wind_mean",
            "f4",
            ("wind_component",),
            np.asarray(stage1["mean"]),
            dict(long_name="Wind normalization mean from Stage-1 training", units="m s-1"),
        )
        variable(
            "input_wind_std",
            "f4",
            ("wind_component",),
            np.asarray(stage1["std"]),
            dict(
                long_name="Wind normalization standard deviation from Stage-1 training",
                units="m s-1",
            ),
        )
        variable(
            "skipped_time",
            "f8",
            ("skipped_time",),
            metadata["times"][metadata["skipped_ids"]].astype("f8") / 1e9,
            dict(
                standard_name="time",
                units=TIME_UNITS,
                calendar=TIME_CALENDAR,
                long_name="Requested targets skipped because of incomplete historical wind",
            ),
        )
        variable(
            "skipped_source_index",
            "i8",
            ("skipped_time",),
            metadata["skipped_ids"],
            dict(
                long_name="Zero-based source indices of skipped target times",
                comment="Insufficient, nonconsecutive, or unavailable historical wind.",
            ),
        )

        for key, (shape, dtype, units) in specs.items():
            name, dims, description = NC_FIELDS[key]
            if tuple(len(out.dimensions[d]) for d in dims) != tuple(shape):
                raise ValueError("NetCDF dimensions do not match output shape: " + name)
            nc_dtype = "i1" if dtype == "?" else np.dtype(dtype)
            fill = np.int8(-1) if dtype == "?" else (-1 if dtype == "i4" else np.nan)
            chunks = (1, 1, h, w) if len(shape) == 4 else (1, h, w)
            var = out.createVariable(
                name,
                nc_dtype,
                dims,
                fill_value=fill,
                chunksizes=chunks,
                zlib=args.compression == "zlib",
                complevel=args.compression_level,
                shuffle=True,
            )
            var.setncatts(dict(long_name=description, units=units))
            if key in ("stage1/hs_experts", "stage2/hs_selected", "stage3/mu"):
                var.standard_name = "sea_surface_wave_significant_height"
            if key == "stage2/expert_index":
                var.valid_range = np.asarray([1, arch["num_experts"]], dtype="i4")
                var.comment = (
                    "Indices map to all_expert and all_expert_name; -1 denotes land or invalid."
                )
            if key == "stage3/mu":
                var.comment = "Gaussian predictive mean; not clipped."
            if key == "stage3/sigma":
                var.comment = "Standard deviation of the Gaussian predictive distribution."
        out.sync()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--era5-dir", required=True, help="Directory containing hourly ERA5 wind NetCDF files."
    )
    parser.add_argument(
        "--file-pattern", default="*.nc", help="Recursive NetCDF filename pattern (default: *.nc)."
    )
    parser.add_argument(
        "--time-var", default="", help="Time variable override (default: auto-detect)."
    )
    parser.add_argument(
        "--lat-var", default="", help="Latitude variable override (default: auto-detect)."
    )
    parser.add_argument(
        "--lon-var", default="", help="Longitude variable override (default: auto-detect)."
    )
    parser.add_argument(
        "--u10-var", default="", help="u10 variable override (default: auto-detect)."
    )
    parser.add_argument(
        "--v10-var", default="", help="v10 variable override (default: auto-detect)."
    )
    parser.add_argument("--static-npz", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", "--selector-checkpoint", default="")
    parser.add_argument("--stage3-checkpoint", "--probability-checkpoint", default="")
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        choices=(1, 2, 3),
        default=[3],
        help="Output stages (default: 3); use 1 2 3 to save all stages.",
    )
    parser.add_argument(
        "--expert-ids",
        nargs="+",
        type=int,
        help="Stage-1 expert IDs to export (one-based; default: all).",
    )
    parser.add_argument("--save-z", action="store_true")
    parser.add_argument("--save-selection-probabilities", action="store_true")
    parser.add_argument("--year", type=int)
    parser.add_argument(
        "--start", help="First target time in UTC, inclusive (ISO date or datetime)."
    )
    parser.add_argument("--end", help="Last target time in UTC, exclusive (ISO date or datetime).")
    parser.add_argument("--invalid-history", choices=("skip", "error"), default="skip")
    parser.add_argument(
        "--max-open-files",
        type=int,
        default=4,
        help="Maximum cached NetCDF handles per data-loading process.",
    )
    parser.add_argument("--output-nc", "--output", required=True, help="Output NetCDF4 file.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda:0 / cuda:1")
    parser.add_argument(
        "--amp", action="store_true", help="Use CUDA float16 inference (default: float32)."
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="NetCDF output precision (default: float32).",
    )
    parser.add_argument("--compression", choices=("zlib", "none"), default="zlib")
    parser.add_argument("--compression-level", type=int, choices=range(1, 10), default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.stages = sorted(set(args.stages))
    if Path(args.output_nc).suffix.lower() not in (".nc", ".nc4"):
        parser.error("--output-nc requires a .nc or .nc4 suffix.")
    if args.batch_size < 1 or args.workers < 0 or args.max_open_files < 1:
        parser.error("batch-size and max-open-files must be positive; workers must be nonnegative.")
    if args.year is not None and (not 1 <= args.year <= 9998 or args.start or args.end):
        parser.error(
            "--year must lie between 1 and 9998 and cannot be combined with --start/--end."
        )
    if args.end and not args.start:
        parser.error("--end requires --start; alternatively use --year.")
    if max(args.stages) == 3 and not args.stage3_checkpoint:
        parser.error("Stage-3 output requires --stage3-checkpoint.")
    if max(args.stages) == 2 and not args.stage2_checkpoint:
        parser.error("Stage-2 output requires --stage2-checkpoint.")
    if args.save_z and 1 not in args.stages:
        parser.error("--save-z requires stage 1 in --stages.")
    if args.save_selection_probabilities and 2 not in args.stages:
        parser.error("--save-selection-probabilities requires stage 2 in --stages.")
    if args.expert_ids is not None and (
        1 not in args.stages or len(set(args.expert_ids)) != len(args.expert_ids)
    ):
        parser.error("--expert-ids requires stage 1 and unique expert IDs.")
    return args


def main(argv=None):
    args = parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "Use python for this single-process exporter; multi-process torchrun is unsupported."
        )
    device = torch.device(
        ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA; omit it for CPU inference.")
    stage1 = load_checkpoint(args.stage1_checkpoint)
    required = {"model", "architecture", "mean", "std", "ocean_mask", "expert_names"}
    if not required.issubset(stage1) or stage1["architecture"].get("version") != 2:
        raise ValueError("Expected a compatible Stage-1 v2 training checkpoint.")
    arch = stage1["architecture"]
    for name in ("mean", "std"):
        value = np.asarray(stage1[name])
        if (
            value.shape != (arch["wind_channels"],)
            or not np.isfinite(value).all()
            or (name == "std" and np.any(value <= 0))
        ):
            raise ValueError("Invalid Stage-1 normalization statistic: " + name)
    names = list(stage1["expert_names"])
    if len(names) != arch["num_experts"] or len(set(names)) != len(names):
        raise ValueError("Invalid Stage-1 expert names or count.")
    expert_ids = (
        [i - 1 for i in args.expert_ids] if args.expert_ids else list(range(arch["num_experts"]))
    )
    if any(i < 0 or i >= arch["num_experts"] for i in expert_ids):
        raise ValueError(f"Expert IDs must lie between 1 and {arch['num_experts']}.")
    dest = Path(args.output_nc).expanduser().resolve()
    partial = Path(str(dest) + ".partial")
    inputs = [
        args.static_npz,
        args.stage1_checkpoint,
        args.stage2_checkpoint,
        args.stage3_checkpoint,
    ]
    if any(
        path
        and (
            dest == Path(path).expanduser().resolve()
            or partial == Path(path).expanduser().resolve()
        )
        for path in inputs
    ):
        raise ValueError("Output path must differ from static inputs and checkpoints.")

    metadata = read_era5_metadata(args, stage1)
    checkpoints = {"stage1": args.stage1_checkpoint}
    if max(args.stages) == 3:
        checkpoints["stage3"] = args.stage3_checkpoint
        if args.stage2_checkpoint:
            checkpoints["stage2"] = args.stage2_checkpoint
    elif max(args.stages) == 2:
        checkpoints["stage2"] = args.stage2_checkpoint
    hashes = {stage: sha256_file(path) for stage, path in checkpoints.items()}
    model = restore_models(args, stage1, metadata, device, hashes)
    count = len(metadata["ids"])
    h, w = metadata["ocean"].shape
    specs = output_specs(args, arch, count, h, w, len(expert_ids))
    estimated = sum(
        np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize
        for shape, dtype, _ in specs.values()
    )
    stamps = time_text(metadata["times"][metadata["ids"][[0, -1]]])
    print(
        f"Device={device}; history={arch['history_hours']} hours; experts={arch['num_experts']}; export stages={args.stages}",
        flush=True,
    )
    print(
        f"ERA5 files={len(metadata['file_info'])}; indexed hourly records={len(metadata['times'])}; file pattern={args.file_pattern}",
        flush=True,
    )
    print(
        f"Targets: requested={metadata['requested_count']}, valid={count}, skipped_incomplete_history={len(metadata['skipped_ids'])}",
        flush=True,
    )
    print(f"UTC: {stamps[0]} to {stamps[1]}; selection={metadata['split']}", flush=True)
    print(f"NetCDF variables: {', '.join(NC_FIELDS[key][0] for key in specs)}", flush=True)
    print(f"Uncompressed arrays: {estimated / 1024**3:.3f} GiB; output={dest}", flush=True)
    if args.dry_run:
        return

    signature_data = dict(
        version=FORMAT_VERSION,
        source_root=metadata["source_root"],
        source_manifest_sha256=metadata["source_manifest_hash"],
        selected=array_hash(
            metadata["ids"], metadata["times"][metadata["ids"]], metadata["skipped_ids"]
        ),
        geometry=array_hash(metadata["geometry"]),
        normalization=array_hash(stage1["mean"], stage1["std"]),
        checkpoint_hashes=hashes,
        stages=args.stages,
        expert_ids=expert_ids,
        fields=specs,
        amp=args.amp,
        dtype=args.dtype,
        compression=args.compression,
        compression_level=args.compression_level,
        output_format="NETCDF4",
        requested_count=metadata["requested_count"],
        selection=metadata["split"],
        file_pattern=args.file_pattern,
        time_var=args.time_var,
        lat_var=args.lat_var,
        lon_var=args.lon_var,
        u10_var=args.u10_var,
        v10_var=args.v10_var,
    )
    signature = json.dumps(signature_data, sort_keys=True)
    if dest.exists():
        if args.resume:
            with netCDF4.Dataset(str(dest), "r") as out:
                out.set_auto_mask(False)
                if (
                    getattr(out, "run_signature", None) == signature
                    and getattr(out, "inference_complete", 0) == 1
                    and np.all(out.variables["completed"][:] == 1)
                ):
                    print("All requested outputs are already complete.", flush=True)
                    return
        raise FileExistsError(f"Output already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        if not args.resume:
            raise FileExistsError(
                f"Partial output exists: {partial}; add --resume or choose another output path."
            )
        with netCDF4.Dataset(str(partial), "r") as out:
            out.set_auto_mask(False)
            if getattr(out, "run_signature", None) != signature:
                raise ValueError("Resume inputs or checkpoints differ from the partial output.")
            if not np.array_equal(out.variables["source_index"][:], metadata["ids"]):
                raise ValueError("Source indices differ from the partial output.")
            for key, spec in specs.items():
                name, dims, _ = NC_FIELDS[key]
                if (
                    name not in out.variables
                    or out.variables[name].shape != spec[0]
                    or out.variables[name].dimensions != dims
                ):
                    raise ValueError("Partial output field or dimensions differ: " + name)
    else:
        initialize_output(partial, signature, specs, metadata, stage1, args, expert_ids)

    dataset = None
    try:
        with netCDF4.Dataset(str(partial), "r+") as out:
            out.set_auto_mask(False)
            rows = np.flatnonzero(out.variables["completed"][:] != 1)
            dataset = InferenceDataset(rows, metadata, stage1, max_open_files=args.max_open_files)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
            )
            with torch.inference_mode():
                for batch in tqdm(loader, desc="WavePrism ERA5 backtest inference", unit="batch"):
                    wind = batch["wind"].to(device, non_blocking=True)
                    with autocast_context(device.type, args.amp):
                        values = predict_batch(model, wind, metadata, args, expert_ids)
                    output_rows = batch["row"].numpy()
                    for key, array in values.items():
                        name = NC_FIELDS[key][0]
                        out.variables[name][output_rows, ...] = array
                    out.sync()
                    out.variables["completed"][output_rows] = np.int8(1)
                    out.sync()
            if not np.all(out.variables["completed"][:] == 1):
                raise RuntimeError("Samples remain incomplete; retaining partial output.")
            out.inference_complete = np.int8(1)
            out.sync()
    finally:
        if dataset is not None:
            dataset.reader.close()
    if dest.exists():
        raise FileExistsError(
            f"Destination appeared during inference; retaining partial output: {partial}"
        )
    os.replace(partial, dest)
    print(f"Saved {count} target times: {dest}", flush=True)


if __name__ == "__main__":
    main()
