from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


def autocast_context(device_type, enabled):
    """Use FP32 on CPU and optional mixed precision on CUDA."""
    return (
        torch.autocast(device_type="cuda", enabled=enabled)
        if device_type == "cuda"
        else contextlib.nullcontext()
    )


def checkpoint_decoder(decoder, z, coarse):
    return activation_checkpoint(decoder, z, coarse, use_reentrant=False)


class GlobalConv(nn.Module):
    def __init__(self, cin, cout, kernel=3, dilation=1, bias=False):
        super().__init__()
        self.pad = dilation * (kernel // 2)
        self.conv = nn.Conv2d(cin, cout, kernel, dilation=dilation, bias=bias)

    def forward(self, x):
        if self.pad:
            x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, self.pad, self.pad), mode="replicate")
        return self.conv(x)


class Stage1ResidualConvBlock(nn.Module):
    def __init__(self, cin, cout, dropout):
        super().__init__()
        self.proj = nn.Identity() if cin == cout else GlobalConv(cin, cout, 1)
        self.net = nn.Sequential(
            GlobalConv(cin, cout),
            nn.GroupNorm(math.gcd(8, cout), cout),
            nn.GELU(),
            nn.Dropout2d(dropout),
            GlobalConv(cout, cout),
            nn.GroupNorm(math.gcd(8, cout), cout),
        )

    def forward(self, x):
        return F.gelu(self.proj(x) + self.net(x))


class Stage1MultiScaleContextBlock(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        branch = max(1, channels // 4)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    GlobalConv(channels, branch, kernel, dilation),
                    nn.GroupNorm(math.gcd(8, branch), branch),
                    nn.GELU(),
                )
                for kernel, dilation in [(1, 1), (3, 1), (3, 2), (3, 4)]
            ]
        )
        self.fuse = nn.Sequential(
            GlobalConv(4 * branch, channels, 1),
            nn.GroupNorm(math.gcd(8, channels), channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class TemporalWindEncoder(nn.Module):
    """Each coarse cell is a chronological series, not a flat channel vector."""

    def __init__(self, wind_channels, channels=16):
        super().__init__()
        self.stem = nn.Conv1d(wind_channels + 1, channels, 5, stride=3, padding=2)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, channels, 3, padding=d, dilation=d),
                    nn.GELU(),
                    nn.Conv1d(channels, channels, 1),
                )
                for d in [1, 2, 4, 8, 16]
            ]
        )
        self.score = nn.Conv1d(channels, 1, 1)
        self.project = nn.Linear(2 * channels, channels)

    def forward(self, wind):
        b, t, c, h, w = wind.shape
        hc, wc = math.ceil(h / 4), math.ceil(w / 4)
        # Pool raw wind first: do NOT allocate T high-resolution CNN activations.
        coarse = F.adaptive_avg_pool2d(wind.reshape(b * t, c, h, w), (hc, wc))
        seq = coarse.reshape(b, t, c, hc, wc).permute(0, 3, 4, 2, 1).reshape(-1, c, t)
        age = torch.linspace(-1.0, 0.0, t, device=wind.device, dtype=wind.dtype)
        seq = torch.cat([seq, age[None, None].expand(seq.shape[0], 1, -1)], 1)
        seq = F.gelu(self.stem(seq))
        for block in self.blocks:
            seq = F.gelu(seq + block(seq))
        weights = self.score(seq).float().softmax(-1).to(seq.dtype)
        summary = torch.cat([(seq * weights).sum(-1), seq[:, :, -1]], 1)
        out = self.project(summary)
        return out.reshape(b, hc, wc, -1).permute(0, 3, 1, 2).contiguous()


class GlobalAttentionBlock(nn.Module):
    """All coarse tokens communicate; cost is bounded by a 1/16 grid."""

    def __init__(self, channels, heads=4, dropout=0.05):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.output = nn.Linear(channels, channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, channels),
        )
        self.heads, self.dropout = heads, dropout
        if channels % heads:
            raise ValueError("Attention channels must be divisible by number of heads.")

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        qkv = self.qkv(self.norm1(tokens)).reshape(b, h * w, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attention = attention.transpose(1, 2).reshape(b, h * w, c)
        tokens = tokens + self.output(attention)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


def periodic_resize(x, size):
    # Longitude wrap is maintained during interpolating coarse features.
    h, w = size
    padded = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)
    yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=torch.float32)
    xx = torch.arange(w, device=x.device, dtype=torch.float32) * (x.shape[-1] / w) + 1
    xx = 2 * xx / (padded.shape[-1] - 1) - 1
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack([gx, gy], -1)[None].expand(x.shape[0], -1, -1, -1)
    # float32 sampling also works with mixed-precision feature tensors.
    return F.grid_sample(
        padded.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
    ).to(x.dtype)


class IndependentExpertDecoder(nn.Module):
    def __init__(self, latent, decoder, dropout):
        super().__init__()
        self.coarse = nn.Sequential(
            Stage1ResidualConvBlock(latent, decoder, dropout),
            Stage1MultiScaleContextBlock(decoder, dropout),
            Stage1ResidualConvBlock(decoder, decoder, dropout),
            GlobalConv(decoder, 1, 1, bias=True),
        )
        self.local = nn.Sequential(
            GlobalConv(latent, 8, 1),
            nn.GELU(),
            GlobalConv(8, 8),
            nn.GELU(),
            GlobalConv(8, 1, 1, bias=True),
        )

    def forward(self, z, zcoarse):
        low = periodic_resize(self.coarse(zcoarse), z.shape[-2:])
        return F.softplus(low + self.local(z))


class WaveExpertModel(nn.Module):
    def __init__(
        self,
        history_hours,
        wind_channels,
        static_channels,
        num_experts,
        latent_channels=48,
        decoder_channels=32,
        dropout=0.05,
        temporal_channels=16,
        checkpoint_decoders=True,
        version=2,
    ):
        super().__init__()
        if version != 2:
            raise ValueError("Only v2 architecture is supported.")
        self.history_hours, self.wind_channels = history_hours, wind_channels
        self.checkpoint_decoders = checkpoint_decoders
        self.temporal = TemporalWindEncoder(wind_channels, temporal_channels)
        g = static_channels  # static geophysics + five position channels + sea mask
        ch = latent_channels
        self.level0 = Stage1ResidualConvBlock(temporal_channels + g, ch, dropout)
        self.down1 = nn.Sequential(
            GlobalConv(ch, ch), nn.GELU(), Stage1ResidualConvBlock(ch, ch, dropout)
        )
        self.down2 = nn.Sequential(
            GlobalConv(ch, ch), nn.GELU(), Stage1ResidualConvBlock(ch, ch, dropout)
        )
        self.geometry = GlobalConv(g, ch, 1)
        self.global_blocks = nn.Sequential(
            GlobalAttentionBlock(ch, 4, dropout), GlobalAttentionBlock(ch, 4, dropout)
        )
        self.up1 = Stage1ResidualConvBlock(2 * ch, ch, dropout)
        self.up0 = Stage1ResidualConvBlock(2 * ch, ch, dropout)
        self.local = nn.Sequential(GlobalConv(3 * wind_channels + g, 16), nn.GELU())
        self.fuse = nn.Sequential(GlobalConv(ch + 16, ch, 1), nn.GELU())
        self.decoders = nn.ModuleList(
            [IndependentExpertDecoder(ch, decoder_channels, dropout) for _ in range(num_experts)]
        )

    def encode(self, wind, static):
        b, t, c, h, w = wind.shape
        if (t, c) != (self.history_hours, self.wind_channels):
            raise ValueError("Unexpected wind history shape.")
        temporal = self.temporal(wind)
        geometry = F.adaptive_avg_pool2d(static, temporal.shape[-2:])
        level0 = self.level0(torch.cat([temporal, geometry], 1))
        level1 = self.down1(F.avg_pool2d(level0, 2, ceil_mode=True))
        level2 = self.down2(F.avg_pool2d(level1, 2, ceil_mode=True))
        # Location/sea information is embedded explicitly at attention resolution.
        geom = F.adaptive_avg_pool2d(static, level2.shape[-2:])
        level2 = self.global_blocks(level2 + self.geometry(geom))
        up1 = self.up1(torch.cat([level1, periodic_resize(level2, level1.shape[-2:])], 1))
        coarse = self.up0(torch.cat([level0, periodic_resize(up1, level0.shape[-2:])], 1))
        recent = wind[:, -min(24, t) :]
        local_input = torch.cat(
            [wind[:, -1], recent.mean(1), recent.std(1, unbiased=False), static], 1
        )
        local = self.local(local_input)
        z = self.fuse(torch.cat([periodic_resize(coarse, (h, w)), local], 1))
        return z, coarse

    def forward(self, wind, static, return_latent=False):
        z, coarse = self.encode(wind, static)
        predictions = []
        for decoder in self.decoders:
            if self.training and self.checkpoint_decoders:
                pred = checkpoint_decoder(decoder, z, coarse)
            else:
                pred = decoder(z, coarse)
            predictions.append(pred)
        hs = torch.cat(predictions, 1)
        return (hs, z) if return_latent else hs
