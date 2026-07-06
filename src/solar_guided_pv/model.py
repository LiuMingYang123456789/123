"""Neural modules for solar-guided PV power forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    power_input_dim: int = 1
    image_channels: int = 3
    mask_channels: int = 3
    horizon: int = 6
    reverso_dim: int = 64
    reverso_layers: int = 4
    reverso_heads: int = 4
    conv_kernel: int = 31
    cloud_dim: int = 64
    fusion_dim: int = 128
    dropout: float = 0.1


class GatedLongConv(nn.Module):
    """Depthwise gated long-convolution block used in the Reverso branch."""

    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.long_conv = nn.Conv1d(dim, dim, kernel_size, groups=dim)
        self.short_conv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: B, L, D. Left padding keeps the operation causal.
        xt = x.transpose(1, 2)
        long = self.long_conv(F.pad(xt, (self.kernel_size - 1, 0)))
        gate = self.short_conv(xt)
        return F.silu(gate * long).transpose(1, 2)


class DeltaNetLayer(nn.Module):
    """Small DeltaNet-style linear RNN layer.

    It implements the delta-rule state update from Reverso at a compact scale.
    This layer is intentionally explicit over time; PV contexts are typically
    moderate in length for fine-tuning experiments.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.beta = nn.Linear(dim, heads)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        beta = torch.sigmoid(self.beta(x)).unsqueeze(-1).unsqueeze(-1)

        state = x.new_zeros(batch, self.heads, self.head_dim, self.head_dim)
        outputs = []
        eye = torch.eye(self.head_dim, device=x.device, dtype=x.dtype).view(
            1, 1, self.head_dim, self.head_dim
        )
        for idx in range(length):
            ki = F.normalize(k[:, idx], dim=-1)
            qi = q[:, idx].unsqueeze(-1)
            vi = v[:, idx].unsqueeze(-1)
            bi = beta[:, idx]
            kk_t = ki.unsqueeze(-1) @ ki.unsqueeze(-2)
            vk_t = vi @ ki.unsqueeze(-2)
            state = state @ (eye - bi * kk_t) + bi * vk_t
            yi = (state @ qi).squeeze(-1)
            outputs.append(yi.reshape(batch, dim))
        y = torch.stack(outputs, dim=1)
        return self.out(y)


class ReversoBlock(nn.Module):
    def __init__(self, dim: int, heads: int, kernel_size: int, use_delta: bool, dropout: float) -> None:
        super().__init__()
        self.sequence = DeltaNetLayer(dim, heads) if use_delta else GatedLongConv(dim, kernel_size)
        self.seq_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.mlp_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.seq_norm(self.sequence(x))
        x = x + self.mlp_norm(self.mlp(x))
        return x


class ReversoEncoder(nn.Module):
    """History PV sequence encoder inspired by Reverso."""

    def __init__(self, input_dim: int, dim: int, layers: int, heads: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList(
            [
                ReversoBlock(
                    dim=dim,
                    heads=heads,
                    kernel_size=kernel_size,
                    use_delta=bool(i % 2),
                    dropout=dropout,
                )
                for i in range(layers)
            ]
        )
        self.pool = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())

    def forward(self, power: Tensor) -> Tensor:
        # power: B, L or B, L, C
        if power.dim() == 2:
            power = power.unsqueeze(-1)
        min_v = power.amin(dim=1, keepdim=True)
        max_v = power.amax(dim=1, keepdim=True)
        x = (power - min_v) / (max_v - min_v + 1e-6)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.pool(x[:, -1])


class SolarGuidedSCSU(nn.Module):
    """Solar-guided spatial channel selection unit.

    The selection weights are generated from both global cloud features and
    sun-neighborhood weighted features, making branch selection sensitive to
    the physically important region around the sun.
    """

    def __init__(self, channels: int, kernel: int = 3) -> None:
        super().__init__()
        strip = 3 * kernel + 2
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel, padding=kernel // 2, groups=channels),
                nn.Conv2d(channels, channels, (1, strip), padding=(0, strip // 2), groups=channels),
                nn.Conv2d(channels, channels, (strip, 1), padding=(strip // 2, 0), groups=channels),
            ]
        )
        self.weight_net = nn.Sequential(
            nn.Conv2d(2 * channels, 3 * channels, 1),
        )

    def forward(self, x: Tensor, masks: Tensor) -> Tensor:
        # x: B,C,H,W; masks: B,S,H,W
        sun_mask = masks.amax(dim=1, keepdim=True)
        global_stat = F.adaptive_avg_pool2d(x, 1)
        sun_stat = F.adaptive_avg_pool2d(x * sun_mask, 1)
        weights = self.weight_net(torch.cat([global_stat, sun_stat], dim=1))
        weights = weights.view(x.shape[0], 3, x.shape[1], 1, 1).softmax(dim=1)
        branch_features = torch.stack([branch(x) for branch in self.branches], dim=1)
        return (weights * branch_features).sum(dim=1)


class CloudFrequencySelectionUnit(nn.Module):
    """Frequency-domain cloud component selection with solar guidance."""

    def __init__(self, channels: int, groups: int = 4) -> None:
        super().__init__()
        if (2 * channels) % groups != 0:
            raise ValueError("2 * channels must be divisible by groups")
        self.groups = groups
        self.group_conv = nn.Conv2d(2 * channels, 2 * channels, 1, groups=groups)
        self.selector = nn.Conv2d(4 * channels, groups, 1)

    def forward(self, x: Tensor, masks: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        sun_mask = masks.amax(dim=1, keepdim=True)
        x_freq = torch.fft.rfft2(x, norm="ortho")
        sun_freq = torch.fft.rfft2(x * sun_mask, norm="ortho")
        freq_real = torch.cat([x_freq.real, x_freq.imag], dim=1)
        sun_real = torch.cat([sun_freq.real, sun_freq.imag], dim=1)
        logits = self.selector(torch.cat([freq_real, sun_real], dim=1)).softmax(dim=1)

        modulated = self.group_conv(freq_real)
        chunks = torch.chunk(modulated, self.groups, dim=1)
        selected_chunks = [chunk * logits[:, i : i + 1] for i, chunk in enumerate(chunks)]
        selected = torch.cat(selected_chunks, dim=1)
        real, imag = torch.chunk(selected, 2, dim=1)
        restored = torch.fft.irfft2(torch.complex(real, imag), s=(height, width), norm="ortho")
        return restored


class SolarGuidedSFSU(nn.Module):
    """Self-calibrated solar-guided spatial-frequency selection unit."""

    def __init__(self, in_channels: int, out_channels: int, mask_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + mask_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.sc_small = SolarGuidedSCSU(out_channels, kernel=3)
        self.sc_large = SolarGuidedSCSU(out_channels, kernel=5)
        self.freq = CloudFrequencySelectionUnit(out_channels, groups=4)
        self.sun_enhance = nn.Sequential(
            nn.Conv2d(out_channels * mask_channels, out_channels, 3, padding=1, groups=1),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, image: Tensor, masks: Tensor) -> Tensor:
        x = self.stem(torch.cat([image, masks], dim=1))
        spatial = 0.5 * (self.sc_small(x, masks) + self.sc_large(x, masks))
        freq = self.freq(x, masks)
        sun_weighted = x.unsqueeze(1) * masks.unsqueeze(2)
        sun = self.sun_enhance(sun_weighted.flatten(1, 2))
        return self.fuse(torch.cat([x, spatial, freq, sun], dim=1))


class ConvLSTMCell(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(channels + hidden_channels, 4 * hidden_channels, 3, padding=1)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor] | None) -> tuple[Tensor, Tensor]:
        if state is None:
            h = x.new_zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3])
            c = x.new_zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3])
        else:
            h, c = state
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class CloudSequenceEncoder(nn.Module):
    def __init__(self, image_channels: int, mask_channels: int, cloud_dim: int) -> None:
        super().__init__()
        self.frame_encoder = SolarGuidedSFSU(image_channels, cloud_dim, mask_channels)
        self.temporal = ConvLSTMCell(cloud_dim, cloud_dim)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(cloud_dim))

    def forward(self, images: Tensor, masks: Tensor) -> Tensor:
        # images: B,T,C,H,W; masks: B,T,S,H,W
        state = None
        for t in range(images.shape[1]):
            frame = self.frame_encoder(images[:, t], masks[:, t])
            state = self.temporal(frame, state)
        assert state is not None
        return self.head(state[0])


class DynamicFusionHead(nn.Module):
    def __init__(self, power_dim: int, cloud_dim: int, fusion_dim: int, horizon: int, dropout: float) -> None:
        super().__init__()
        self.power_proj = nn.Linear(power_dim, fusion_dim)
        self.cloud_proj = nn.Linear(cloud_dim, fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, horizon),
        )

    def forward(self, power_feat: Tensor, cloud_feat: Tensor) -> Tensor:
        fp = self.power_proj(power_feat)
        fc = self.cloud_proj(cloud_feat)
        alpha = self.gate(torch.cat([fp, fc], dim=-1))
        fused = alpha * fc + (1.0 - alpha) * fp
        return self.out(fused)


class SolarGuidedPVForecaster(nn.Module):
    """Full Reverso + SC-SG-SFSU multimodal forecaster."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.power_encoder = ReversoEncoder(
            input_dim=config.power_input_dim,
            dim=config.reverso_dim,
            layers=config.reverso_layers,
            heads=config.reverso_heads,
            kernel_size=config.conv_kernel,
            dropout=config.dropout,
        )
        self.cloud_encoder = CloudSequenceEncoder(
            image_channels=config.image_channels,
            mask_channels=config.mask_channels,
            cloud_dim=config.cloud_dim,
        )
        self.fusion = DynamicFusionHead(
            power_dim=config.reverso_dim,
            cloud_dim=config.cloud_dim,
            fusion_dim=config.fusion_dim,
            horizon=config.horizon,
            dropout=config.dropout,
        )

    def forward(self, power: Tensor, images: Tensor, masks: Tensor) -> Tensor:
        power_feat = self.power_encoder(power)
        cloud_feat = self.cloud_encoder(images, masks)
        return self.fusion(power_feat, cloud_feat)
