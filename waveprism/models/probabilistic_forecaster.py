from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .expert_selector import ExpertSelector, MultiScaleContextBlock, ResidualConvBlock, Stage2Config


class ProbabilityResidualHead(nn.Module):
    """Predict a mean residual and positive standard deviation for the selected expert."""

    def __init__(
        self, in_ch: int, hidden_ch: int = 96, dropout: float = 0.05, min_sigma: float = 0.05
    ):
        super().__init__()

        self.min_sigma = float(min_sigma)

        self.net = nn.Sequential(
            ResidualConvBlock(in_ch, hidden_ch, dropout=dropout),
            ResidualConvBlock(hidden_ch, hidden_ch, dropout=dropout),
            MultiScaleContextBlock(hidden_ch, hidden_ch, dropout=dropout),
            ResidualConvBlock(hidden_ch, hidden_ch, dropout=dropout),
        )

        self.delta_head = nn.Conv2d(hidden_ch, 1, kernel_size=1)
        self.sigma_head = nn.Conv2d(hidden_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, selected_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(x)

        delta_mu = self.delta_head(h)
        raw_sigma = self.sigma_head(h)

        sigma = F.softplus(raw_sigma) + self.min_sigma
        mu = selected_pred + delta_mu

        return {"mu": mu, "sigma": sigma, "delta_mu": delta_mu, "raw_sigma": raw_sigma}


class SelectorProbabilityModel(nn.Module):
    """Combine the frozen expert forecaster, expert selector, and probability residual head."""

    def __init__(self, cfg: Stage2Config, prior=None):
        super().__init__()
        self.cfg = cfg
        self.selector = ExpertSelector(cfg, prior=prior)

        z_ch = self.selector.wind_encoder.out_channels
        prob_static_ch = cfg.static_channels if cfg.use_static_features else 0
        prob_in_ch = z_ch + 1 + 1 + 1 + 1 + prob_static_ch + cfg.num_experts

        self.prob_head = ProbabilityResidualHead(
            in_ch=prob_in_ch,
            hidden_ch=cfg.prob_hidden_channels,
            dropout=cfg.dropout,
            min_sigma=cfg.min_sigma,
        )

    def forward_selector(
        self,
        x_in: torch.Tensor,
        m_mod: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.selector(x_in, m_mod, static, position)

    def build_prob_input(
        self, selector_outputs: Dict[str, torch.Tensor], static: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.cfg.use_static_features:
            if static is None:
                raise ValueError("static is required for probability head.")
        else:
            b, _, h, w = selector_outputs["selected_pred"].shape
            static = torch.empty((b, 0, h, w), device=selector_outputs["selected_pred"].device)

        return torch.cat(
            [
                selector_outputs["z"],
                selector_outputs["selected_pred"],
                selector_outputs["selector_conf"],
                selector_outputs["m_std"],
                selector_outputs["m_range"],
                static,
                selector_outputs["selected_onehot"],
            ],
            dim=1,
        )

    def forward_probability(
        self,
        x_in: torch.Tensor,
        m_mod: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
        detach_selector: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if detach_selector:
            with torch.no_grad():
                selector_outputs = self.forward_selector(x_in, m_mod, static, position)
        else:
            selector_outputs = self.forward_selector(x_in, m_mod, static, position)

        prob_in = self.build_prob_input(selector_outputs, static)
        prob_outputs = self.prob_head(prob_in, selector_outputs["selected_pred"])
        return {**selector_outputs, **prob_outputs}

    def forward(
        self,
        x_in: torch.Tensor,
        m_mod: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
        stage: str = "prob",
    ) -> Dict[str, torch.Tensor]:
        if stage == "selector":
            return self.forward_selector(x_in, m_mod, static, position)
        if stage == "prob":
            return self.forward_probability(x_in, m_mod, static, position, detach_selector=False)
        raise ValueError(f"Unsupported stage: {stage}")
