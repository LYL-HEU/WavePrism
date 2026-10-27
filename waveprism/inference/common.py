from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from ..models.expert_forecaster import WaveExpertModel
from ..models.expert_selector import Stage2Config
from ..models.factory import get_stage1_context, set_stage1_context
from ..models.probabilistic_forecaster import SelectorProbabilityModel

FORMAT_VERSION = 4


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def plain_state(state):
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("optimizer", "optimizer_state_dict", "scheduler", "scaler", "rank_states"):
        checkpoint.pop(key, None)
    return checkpoint


def utc_ns(value):
    """Date / ISO datetime in UTC. Naive input is interpreted as UTC."""
    from datetime import datetime, timezone

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    delta = parsed - datetime(1970, 1, 1)
    result = (delta.days * 86400 + delta.seconds) * 1000000000 + delta.microseconds * 1000
    if not np.iinfo(np.int64).min < result <= np.iinfo(np.int64).max:
        raise ValueError("Date lies outside the Unix nanosecond timestamp range.")
    return result


def time_text(times):
    return np.datetime_as_string(
        np.asarray(times).astype("datetime64[ns]"), unit="s", timezone="UTC"
    )


def restore_models(args, stage1, metadata, device, hashes):
    """Restore all weights without training data, observations, or prior scans."""
    arch = dict(stage1["architecture"])
    s1state = plain_state(stage1["model"])
    if max(args.stages) == 1:
        model = WaveExpertModel(**arch)
        model.load_state_dict(s1state, strict=True)
    else:
        path = args.stage3_checkpoint if max(args.stages) == 3 else args.stage2_checkpoint
        checkpoint = load_checkpoint(path)
        if max(args.stages) == 3 and checkpoint.get("stage") != "probability":
            raise ValueError("--stage3-checkpoint must contain the trained probability module.")
        if max(args.stages) == 2 and checkpoint.get("stage") != "selector":
            raise ValueError("--stage2-checkpoint must contain the trained selector.")
        saved = checkpoint.get("config", {})
        if saved.get("expert_source") != "stage1_generated_v1":
            raise ValueError("Expected a checkpoint trained on Stage-1 generated experts.")
        if saved.get("stage1_checkpoint_sha256") != hashes["stage1"]:
            raise ValueError(
                "Stage-1 checkpoint hash differs from the downstream training checkpoint."
            )
        if saved.get("stage1_architecture") != arch:
            raise ValueError("Stage architectures differ.")
        state = plain_state(checkpoint["model_state_dict"])
        for key, value in s1state.items():
            stored = state.get("selector.wind_encoder.core." + key)
            if stored is None or not torch.equal(value.cpu(), stored.cpu()):
                raise ValueError("Frozen Stage-1 weight differs: " + key)
        geometry = torch.from_numpy(metadata["geometry"][None])
        if "selector.wind_encoder.geometry" not in state or not torch.equal(
            state["selector.wind_encoder.geometry"].cpu(), geometry
        ):
            raise ValueError("Inference geometry differs from training geometry.")
        if max(args.stages) == 3 and args.stage2_checkpoint:
            previous = load_checkpoint(args.stage2_checkpoint)
            if previous.get("stage") != "selector":
                raise ValueError("--stage2-checkpoint is not a selector checkpoint.")
            previous_state = plain_state(previous["model_state_dict"])
            for key, value in state.items():
                if key.startswith("selector.") and (
                    key not in previous_state
                    or not torch.equal(value.cpu(), previous_state[key].cpu())
                ):
                    raise ValueError("Stage-2 selector differs from Stage-3 checkpoint: " + key)
            del previous, previous_state
        cfg = Stage2Config()
        for key, value in saved.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        if (
            cfg.num_experts != arch["num_experts"]
            or cfg.in_time != arch["history_hours"]
            or cfg.in_channels != arch["wind_channels"]
            or cfg.wind_base_channels != arch["latent_channels"]
        ):
            raise ValueError("Selector expert count, history length, or feature channels differ.")
        cfg.encoder_amp = args.amp
        cfg.stage1_checkpoint = str(Path(args.stage1_checkpoint).resolve())
        # The inference path is ERA5 NetCDF only; the model constructor does not read cfg.data_h5.
        cfg.data_h5, cfg.static_npz = "", args.static_npz
        # These classes already implement the exact training forward path.
        # Restore the TRAIN prior from checkpoint buffers; never recompute it.
        old_context = get_stage1_context(required=False)
        set_stage1_context(
            dict(checkpoint=dict(model=s1state), meta=arch, geometry=metadata["geometry"])
        )
        try:
            prior = None
            if cfg.use_expert_skill_prior:
                prior = tuple(
                    state[key][0].cpu().numpy()
                    for key in (
                        "selector.skill_prior_logits_fixed",
                        "selector.skill_prior_confidence_fixed",
                    )
                )
            model = SelectorProbabilityModel(cfg, prior=prior)
            model.load_state_dict(state, strict=True)
        finally:
            set_stage1_context(old_context)
        del checkpoint, state
    model.requires_grad_(False)
    return model.to(device).eval()


@torch.inference_mode()
def predict_batch(model, wind, metadata, args, expert_ids):
    batch = wind.shape[0]
    geometry = torch.as_tensor(metadata["geometry"], device=wind.device)[None].expand(
        batch, -1, -1, -1
    )
    outputs = {}
    if max(args.stages) == 1:
        z, coarse = model.encode(wind, geometry)
        experts = torch.cat([model.decoders[index](z, coarse) for index in expert_ids], dim=1)
        outputs["stage1/hs_experts"] = experts
    else:
        z, all_experts = model.selector.wind_encoder(wind)
        if 1 in args.stages:
            outputs["stage1/hs_experts"] = all_experts[:, expert_ids]
        static = torch.as_tensor(metadata["static"], device=wind.device)[None].expand(
            batch, -1, -1, -1
        )
        position = torch.as_tensor(metadata["position"], device=wind.device)[None].expand(
            batch, -1, -1, -1
        )
        selected = model.forward_selector(z, all_experts, static=static, position=position)
        if 2 in args.stages:
            outputs.update(
                {
                    "stage2/hs_selected": selected["selected_pred"],
                    "stage2/expert_index": selected["pred_idx"] + 1,
                    "stage2/confidence": selected["selector_conf"],
                    "stage2/expert_std": selected["m_std"],
                    "stage2/expert_range": selected["m_range"],
                }
            )
            if args.save_selection_probabilities:
                outputs["stage2/selection_probability"] = selected["prob"]
        if 3 in args.stages:
            probability = model.prob_head(
                model.build_prob_input(selected, static), selected["selected_pred"]
            )
            outputs.update({"stage3/" + name: probability[name] for name in ("mu", "sigma")})
    if args.save_z:
        outputs["stage1/z"] = z
    ocean = torch.as_tensor(metadata["ocean"], device=wind.device)[None, None]
    for name, value in outputs.items():
        valid = ocean.expand_as(value)
        if name == "stage2/expert_index":
            value = value.masked_fill(~valid, -1)
        else:
            value = value.float()
            if not torch.isfinite(value.masked_select(valid)).all():
                raise FloatingPointError(f"Nonfinite output over ocean: {name}")
            value = value.masked_fill(~valid, float("nan"))
        # Expert and latent arrays retain the channel axis even for one expert.
        if not (name.startswith("stage1/") or name == "stage2/selection_probability"):
            value = value[:, 0]
        outputs[name] = value.cpu().numpy()
    return outputs


def output_specs(args, arch, count, h, w, expert_count):
    specs = {}
    if 1 in args.stages:
        specs["stage1/hs_experts"] = ((count, expert_count, h, w), args.dtype, "m")
    if args.save_z:
        specs["stage1/z"] = ((count, arch["latent_channels"], h, w), args.dtype, "1")
    if 2 in args.stages:
        for name, unit in [
            ("hs_selected", "m"),
            ("confidence", "1"),
            ("expert_std", "m"),
            ("expert_range", "m"),
        ]:
            specs["stage2/" + name] = ((count, h, w), args.dtype, unit)
        specs["stage2/expert_index"] = ((count, h, w), "i4", "1")
        if args.save_selection_probabilities:
            specs["stage2/selection_probability"] = (
                (count, arch["num_experts"], h, w),
                args.dtype,
                "1",
            )
    if 3 in args.stages:
        for name in ("mu", "sigma"):
            specs["stage3/" + name] = ((count, h, w), args.dtype, "m")
    return specs


NC_FIELDS = {
    "stage1/hs_experts": (
        "stage1_hs",
        ("time", "expert", "lat", "lon"),
        "Significant wave height generated by Stage-1 expert decoders",
    ),
    "stage1/z": (
        "stage1_z",
        ("time", "latent_channel", "lat", "lon"),
        "Latent wind features from the frozen Stage-1 encoder",
    ),
    "stage2/hs_selected": (
        "stage2_hs_selected",
        ("time", "lat", "lon"),
        "Significant wave height of the selected expert",
    ),
    "stage2/expert_index": (
        "stage2_expert_index",
        ("time", "lat", "lon"),
        "Selected Stage-1 decoder index, starting at one",
    ),
    "stage2/confidence": (
        "stage2_confidence",
        ("time", "lat", "lon"),
        "Selection probability assigned to the selected expert",
    ),
    "stage2/expert_std": (
        "stage2_expert_std",
        ("time", "lat", "lon"),
        "Standard deviation across all Stage-1 expert predictions",
    ),
    "stage2/expert_range": (
        "stage2_expert_range",
        ("time", "lat", "lon"),
        "Maximum minus minimum across all Stage-1 expert predictions",
    ),
    "stage2/selection_probability": (
        "stage2_selection_probability",
        ("time", "all_expert", "lat", "lon"),
        "Selection probability for each Stage-1 expert",
    ),
    "stage3/mu": (
        "stage3_mu",
        ("time", "lat", "lon"),
        "Predictive mean of significant wave height",
    ),
    "stage3/sigma": (
        "stage3_sigma",
        ("time", "lat", "lon"),
        "Predictive standard deviation of significant wave height",
    ),
}

TIME_UNITS = "seconds since 1970-01-01 00:00:00"

TIME_CALENDAR = "proleptic_gregorian"
