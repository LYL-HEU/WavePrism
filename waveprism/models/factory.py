"""Model construction and shared Stage-1 runtime context."""

from __future__ import annotations

from typing import Any, Mapping

_STAGE1_CONTEXT = None


def set_stage1_context(context):
    """Set the process-local frozen Stage-1 context used by selector construction."""
    global _STAGE1_CONTEXT
    _STAGE1_CONTEXT = context


def get_stage1_context(required: bool = True):
    """Return the current Stage-1 context or raise when model construction needs it."""
    if required and _STAGE1_CONTEXT is None:
        raise RuntimeError(
            "Stage-1 context has not been prepared. Call prepare_stage1_context first."
        )
    return _STAGE1_CONTEXT


def build_expert_forecaster(architecture: Mapping[str, Any], state_dict=None, strict: bool = True):
    """Construct the Stage-1 expert model and optionally load its state dictionary."""
    from .expert_forecaster import WaveExpertModel

    model = WaveExpertModel(**dict(architecture))
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=strict)
    return model


def build_probabilistic_forecaster(config, stage1_context=None):
    """Construct the selector-plus-probability model from a prepared config."""
    from .probabilistic_forecaster import SelectorProbabilityModel

    if stage1_context is not None:
        set_stage1_context(stage1_context)
    return SelectorProbabilityModel(config)
