# -*- coding: utf-8 -*-
"""Self-contained SpectraSafe anonymizer described in the paper.

Input and output are RGB videos with shape ``[B, 3, T, H, W]`` in ``[0, 1]``.
The default configuration follows the paper specification:

1. RGB -> centered YUV.
2. Spatial branch: 16x16 block DCT, one complete-frequency token per patch,
   frame-local 2-D patch attention, then per-DCT-point G and sigma.
3. Temporal branch: temporal rFFT of the DCT-reconstructed YUV video, one
   complex-spectrum token per patch and frequency bin, then attention across
   the nine temporal frequency bins inside each spatial patch. Patch context
   and per-coefficient real/imaginary YUV values jointly predict every G and
   sigma coefficient.
4. A detached skin/motion guidance map modulates both policy branches.
5. Both branches apply bounded suppression and Laplace replacement noise in
   frequency space, followed by IDCT/irFFT.

There are no dataset, classifier, training-loop, Mamba, or dynamic-import
dependencies in this file. Checkpoints from historical full-FFT experiments
are intentionally incompatible with this paper-spec implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass(frozen=True)
class FAConfig:
    """Architecture and frequency-policy settings for SpectraSafe."""

    num_frames: int = 16
    image_size: int = 224
    block_size: int = 16
    num_blocks: int = 1
    seed: int = 42

    hidden_dim: int = 256
    mlp_depth: int = 2
    mlp_dropout: float = 0.0
    dct_attention_heads: int = 8
    dct_attention_dropout: float = 0.0
    dct_context_init_scale: float = 0.05
    dct_context_max_scale: float = 0.10
    fft_attention_heads: int = 8
    fft_attention_layers: int = 1
    fft_attention_dropout: float = 0.0
    fft_context_init_scale: float = 0.05
    fft_context_max_scale: float = 0.10
    fft_point_hidden_dim: int = 64
    policy_bf16: bool = True

    fft_norm: str = "ortho"
    gate_min: float = 0.001
    gate_max: float = 0.999
    dct_sigma_max: float = 0.42
    fft_sigma_max: float = 0.28
    dct_noise_cap_ratio: float = 0.50
    fft_noise_cap_ratio: float = 0.18
    dct_clip_quantile: float = 0.99
    fft_clip_quantile: float = 0.99
    clip_min: float = 1.0e-4

    mask_skin_weight: float = 1.0
    mask_motion_weight: float = 1.0
    mask_luma_center: float = 0.56
    mask_luma_sigma: float = 0.32
    mask_cb_center: float = 0.42
    mask_cb_sigma: float = 0.105
    mask_cr_center: float = 0.60
    mask_cr_sigma: float = 0.105
    mask_blur_kernel: int = 9
    mask_power: float = 0.6
    dct_mask_gate_drop: float = 0.82
    dct_mask_noise_gain: float = 0.18
    fft_mask_gate_drop: float = 0.35
    fft_mask_noise_gain: float = 0.055

    dct_low_radius: float = 0.22
    dct_mid_radius: float = 0.55
    temporal_low_ratio: float = 0.25
    temporal_mid_ratio: float = 0.65
    dct_uv_gate_init_scale: float = 0.80

    # Tuple order is (DC, low, mid, high). These values initialize trainable
    # policy-head biases; they are not fixed residual tables during forward.
    y_dct_gate_init: Tuple[float, ...] = (0.98, 0.80, 0.34, 0.12)
    uv_dct_gate_init: Tuple[float, ...] = (0.28, 0.16, 0.055, 0.025)
    y_dct_sigma_init: Tuple[float, ...] = (0.0, 0.02, 0.08, 0.07)
    uv_dct_sigma_init: Tuple[float, ...] = (0.09, 0.13, 0.17, 0.14)

    y_fft_gate_init: Tuple[float, ...] = (0.96, 0.82, 0.56, 0.42)
    uv_fft_gate_init: Tuple[float, ...] = (0.16, 0.10, 0.06, 0.04)
    y_fft_sigma_init: Tuple[float, ...] = (0.0, 0.014, 0.04, 0.035)
    uv_fft_sigma_init: Tuple[float, ...] = (0.06, 0.10, 0.11, 0.085)

    # Guidance profiles remain fixed. They describe where the guidance map has
    # more or less influence; Attention still predicts every G and sigma.
    y_dct_guidance: Tuple[float, ...] = (0.12, 0.75, 1.00, 0.90)
    uv_dct_guidance: Tuple[float, ...] = (0.85, 1.25, 1.20, 1.00)
    y_fft_guidance: Tuple[float, ...] = (0.20, 0.35, 0.70, 0.75)
    uv_fft_guidance: Tuple[float, ...] = (0.75, 0.90, 1.00, 1.00)


# ---------------------------------------------------------------------------
# Color conversion and fixed frequency transforms
# ---------------------------------------------------------------------------


def rgb_to_centered_yuv(rgb: Tensor) -> Tensor:
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299000 * r + 0.587000 * g + 0.114000 * b
    cb = 0.500000 - 0.168736 * r - 0.331264 * g + 0.500000 * b
    cr = 0.500000 + 0.500000 * r - 0.418688 * g - 0.081312 * b
    return torch.cat([y, cb - 0.5, cr - 0.5], dim=1)


def centered_yuv_to_rgb(yuv: Tensor) -> Tensor:
    y = yuv[:, 0:1]
    cb = yuv[:, 1:2] + 0.5
    cr = yuv[:, 2:3] + 0.5
    u, v = cb - 0.5, cr - 0.5
    r = y + 1.402000 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772000 * u
    return torch.cat([r, g, b], dim=1)


def make_dct_matrix(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    frequency = torch.arange(size, device=device, dtype=dtype).view(size, 1)
    sample = torch.arange(size, device=device, dtype=dtype).view(1, size)
    matrix = torch.cos(math.pi / size * (sample + 0.5) * frequency)
    matrix[0] *= math.sqrt(1.0 / size)
    if size > 1:
        matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


def pad_to_blocks(x: Tensor, block_size: int) -> Tuple[Tensor, Tuple[int, int]]:
    height, width = x.shape[-2:]
    pad_h = (block_size - height % block_size) % block_size
    pad_w = (block_size - width % block_size) % block_size
    if pad_h == 0 and pad_w == 0:
        return x, (height, width)
    return F.pad(x, (0, pad_w, 0, pad_h, 0, 0), mode="reflect"), (height, width)


def blockify(x: Tensor, block_size: int) -> Tensor:
    batch, channels, groups, height, width = x.shape
    rows, cols = height // block_size, width // block_size
    return (
        x.view(batch, channels, groups, rows, block_size, cols, block_size)
        .permute(0, 1, 2, 3, 5, 4, 6)
        .contiguous()
    )


def unblockify(blocks: Tensor) -> Tensor:
    batch, channels, groups, rows, cols, block_h, block_w = blocks.shape
    return (
        blocks.permute(0, 1, 2, 3, 5, 4, 6)
        .contiguous()
        .view(batch, channels, groups, rows * block_h, cols * block_w)
    )


def block_dct2(x: Tensor, block_size: int, matrix: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
    padded, original_hw = pad_to_blocks(x, block_size)
    blocks = blockify(padded, block_size)
    flat = blocks.view(-1, block_size, block_size)
    coeff = matrix @ flat @ matrix.t()
    return coeff.view_as(blocks), original_hw


def block_idct2(coeff: Tensor, original_hw: Tuple[int, int], matrix: Tensor) -> Tensor:
    block_size = coeff.shape[-1]
    flat = coeff.view(-1, block_size, block_size)
    blocks = matrix.t() @ flat @ matrix
    reconstructed = unblockify(blocks.view_as(coeff))
    height, width = original_hw
    return reconstructed[..., :height, :width].contiguous()


def clip_dct_coefficients(coeff: Tensor, quantile: float, minimum: float) -> Tuple[Tensor, Tensor]:
    magnitude = coeff.abs()
    batch, channels, _, _, _, block_h, block_w = magnitude.shape
    values = (
        magnitude.permute(0, 1, 5, 6, 2, 3, 4)
        .contiguous()
        .flatten(4)
    )
    if quantile >= 1.0:
        clip = values.amax(dim=4)
    else:
        clip = torch.quantile(values.float(), q=float(quantile), dim=4).to(magnitude.dtype)
    clip = clip.clamp_min(float(minimum))
    limit = clip.view(batch, channels, 1, 1, 1, block_h, block_w)
    scale = (limit / (magnitude + 1.0e-12)).clamp(max=1.0)
    return coeff * scale, clip


def clip_complex_spectrum(z: Tensor, quantile: float, minimum: float) -> Tuple[Tensor, Tensor]:
    magnitude = z.abs()
    flat = magnitude.flatten(3)
    if quantile >= 1.0:
        clip = flat.amax(dim=3)
    else:
        clip = torch.quantile(flat.float(), q=float(quantile), dim=3).to(magnitude.dtype)
    clip = clip.view(z.size(0), z.size(1), z.size(2), 1, 1).clamp_min(float(minimum))
    return z * (clip / (magnitude + 1.0e-12)).clamp(max=1.0), clip


def sample_laplace(shape, device: torch.device, dtype: torch.dtype) -> Tensor:
    uniform = torch.rand(shape, device=device, dtype=dtype) - 0.5
    return -torch.sign(uniform) * torch.log((1.0 - 2.0 * uniform.abs()).clamp_min(1.0e-12))


def sample_complex_laplace(z: Tensor, scale: Tensor, original_frames: int) -> Tensor:
    real = sample_laplace(z.shape, z.device, z.real.dtype) * scale
    imag = sample_laplace(z.shape, z.device, z.real.dtype) * scale
    imag[:, :, 0] = 0.0
    if original_frames % 2 == 0 and z.size(2) > 1:
        imag[:, :, -1] = 0.0
    return torch.complex(real, imag)


# ---------------------------------------------------------------------------
# Policy networks
# ---------------------------------------------------------------------------


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(probability / (1.0 - probability))


def _bounded_logit(value: Tensor, low: float, high: float) -> Tensor:
    probability = ((value.float() - low) / max(high - low, 1.0e-8)).clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.log(probability / (1.0 - probability))


class PatchPolicyMLP(nn.Module):
    """Local complete-frequency MLP used before DCT spatial attention."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int, dropout: float):
        super().__init__()
        layers = []
        current = int(in_dim)
        for _ in range(max(1, int(depth))):
            layers.extend([nn.Linear(current, hidden_dim), nn.GELU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = hidden_dim
        layers.append(nn.Linear(current, out_dim))
        self.net = nn.Sequential(*layers)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, rows, cols, width = x.shape
        return self.net(x.reshape(batch * rows * cols, width)).view(batch, rows, cols, -1)


class SpatialSelfAttention2D(nn.Module):
    """Self-attention across all spatial patches of one frame."""

    def __init__(self, hidden_dim: int, rows: int, cols: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.rows = rows
        self.cols = cols
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        batch, rows, cols, dim = hidden.shape
        if (rows, cols, dim) != (self.rows, self.cols, self.hidden_dim):
            raise ValueError(f"DCT attention grid mismatch: {(rows, cols, dim)}")
        length = rows * cols
        q, k, v = self.qkv(self.input_norm(hidden).reshape(batch, length, dim)).chunk(3, dim=-1)

        def heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        context = F.scaled_dot_product_attention(
            heads(q), heads(k), heads(v),
            dropout_p=self.dropout if self.training else 0.0,
        )
        context = context.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.output_proj(context).view(batch, rows, cols, dim)


class ResidualFrequencyAttentionPolicy(nn.Module):
    """Complete-frequency MLP plus a bounded cross-patch residual."""

    def __init__(self, baseline: PatchPolicyMLP, hidden_dim: int, rows: int, cols: int, config: FAConfig, seed: int):
        super().__init__()
        layers = list(baseline.net.children())
        self.trunk = nn.Sequential(*layers[:-1])
        self.policy_head = layers[-1]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.context = SpatialSelfAttention2D(
                hidden_dim, rows, cols,
                config.dct_attention_heads,
                config.dct_attention_dropout,
            )
        self.context_max_scale = config.dct_context_max_scale
        ratio = config.dct_context_init_scale / config.dct_context_max_scale
        self.context_scale_logit = nn.Parameter(torch.tensor(_logit(ratio)))

    def context_scale(self) -> Tensor:
        return self.context_max_scale * torch.sigmoid(self.context_scale_logit)

    def forward(self, tokens: Tensor) -> Tensor:
        hidden = self.trunk(tokens)
        context = self.context(hidden)
        local_rms = torch.sqrt(hidden.float().square().mean(dim=-1, keepdim=True) + 1.0e-8).detach().to(hidden.dtype)
        correction = self.context_scale().to(hidden.dtype) * local_rms * torch.tanh(context)
        return self.policy_head(hidden + correction)


class ResidualBandAttention(nn.Module):
    """Bounded self-attention across temporal frequency bins in one patch."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, init_scale: float, max_scale: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.max_scale = max_scale
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self.scale_logit = nn.Parameter(torch.tensor(_logit(init_scale / max_scale)))

    def context_scale(self) -> Tensor:
        return self.max_scale * torch.sigmoid(self.scale_logit)

    def forward(self, sequence: Tensor) -> Tensor:
        batch, length, dim = sequence.shape
        q, k, v = self.qkv(self.input_norm(sequence)).chunk(3, dim=-1)

        def heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        context = F.scaled_dot_product_attention(
            heads(q), heads(k), heads(v),
            dropout_p=self.dropout if self.training else 0.0,
        )
        context = context.transpose(1, 2).contiguous().view(batch, length, dim)
        context = self.output_proj(context)
        local_rms = torch.sqrt(sequence.float().square().mean(dim=-1, keepdim=True) + 1.0e-8).detach().to(sequence.dtype)
        correction = self.context_scale().to(sequence.dtype) * local_rms * torch.tanh(context)
        return sequence + correction


class PatchBandFFTPolicy(nn.Module):
    """Intra-patch rFFT attention with coefficient-wise policy heads.

    The patch token supplies cross-frequency context.  A second shared encoder
    sees the real/imaginary YUV value at every coefficient position, so the
    final gate and noise scale are predicted independently for every
    ``(channel, frequency, y, x)`` coefficient instead of being interpolated
    from six patch-level scalars.
    """

    def __init__(self, config: FAConfig, rows: int, cols: int, gate_init: Tensor, sigma_init: Tensor):
        super().__init__()
        block = config.block_size
        self.block = block
        self.input_dim = 2 * 3 * block * block + 3
        self.freq_bins = config.num_frames // 2 + 1
        self.rows = rows
        self.cols = cols
        self.hidden_dim = config.hidden_dim
        self.point_dim = config.fft_point_hidden_dim
        self.patch_encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.frequency_position = nn.Parameter(torch.zeros(1, 1, 1, self.freq_bins, self.hidden_dim))
        self.row_position = nn.Parameter(torch.zeros(1, self.rows, 1, 1, self.hidden_dim))
        self.col_position = nn.Parameter(torch.zeros(1, 1, self.cols, 1, self.hidden_dim))
        for position in (self.frequency_position, self.row_position, self.col_position):
            nn.init.trunc_normal_(position, std=0.02)
        self.band_attention = nn.ModuleList([
            ResidualBandAttention(
                self.hidden_dim,
                config.fft_attention_heads,
                config.fft_attention_dropout,
                config.fft_context_init_scale,
                config.fft_context_max_scale,
            )
            for _ in range(config.fft_attention_layers)
        ])
        self.context_norm = nn.LayerNorm(self.hidden_dim)
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, self.point_dim),
            nn.GELU(),
            nn.Linear(self.point_dim, self.point_dim),
            nn.GELU(),
        )
        self.context_head = nn.Linear(self.hidden_dim, 6, bias=False)
        self.point_head = nn.Linear(self.point_dim, 6, bias=False)
        nn.init.normal_(self.context_head.weight, mean=0.0, std=1.0e-4)
        nn.init.normal_(self.point_head.weight, mean=0.0, std=1.0e-4)
        fft_bias = torch.cat([
            _bounded_logit(gate_init, config.gate_min, config.gate_max).transpose(0, 1),
            _bounded_logit(sigma_init, 0.0, config.fft_sigma_max).transpose(0, 1),
        ], dim=-1)
        self.band_policy_bias = nn.Parameter(fft_bias.contiguous())
        self.policy_bf16 = config.policy_bf16

    def forward(self, patch_tokens: Tensor, point_tokens: Tensor) -> Tensor:
        batch, rows, cols, frequencies, width = patch_tokens.shape
        expected = (self.rows, self.cols, self.freq_bins, self.input_dim)
        if (rows, cols, frequencies, width) != expected:
            raise ValueError(f"FFT policy token mismatch: {(rows, cols, frequencies, width)} vs {expected}")
        expected_points = (
            batch, rows, cols, frequencies, self.block, self.block, 6
        )
        if tuple(point_tokens.shape) != expected_points:
            raise ValueError(
                f"FFT point token mismatch: {tuple(point_tokens.shape)} vs {expected_points}"
            )
        use_amp = self.policy_bf16 and patch_tokens.is_cuda
        with torch.amp.autocast(patch_tokens.device.type, dtype=torch.bfloat16, enabled=use_amp):
            hidden = (
                self.patch_encoder(patch_tokens)
                + self.frequency_position
                + self.row_position
                + self.col_position
            )
            sequence = hidden.reshape(batch * rows * cols, frequencies, self.hidden_dim)
            for attention in self.band_attention:
                sequence = attention(sequence)
            hidden = sequence.view(batch, rows, cols, frequencies, self.hidden_dim)
            context_logits = self.context_head(self.context_norm(hidden)).unsqueeze(4).unsqueeze(5)
            point_logits = self.point_head(self.point_encoder(point_tokens))
            raw = context_logits + point_logits
        return raw.float() + self.band_policy_bias.view(1, 1, 1, frequencies, 1, 1, 6)


# ---------------------------------------------------------------------------
# Frequency anonymization block
# ---------------------------------------------------------------------------


def _dct_band_masks(config: FAConfig) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    block = config.block_size
    u = torch.arange(block, dtype=torch.float32).view(block, 1)
    v = torch.arange(block, dtype=torch.float32).view(1, block)
    radius = torch.sqrt(u.square() + v.square())
    radius = radius / radius.max().clamp_min(1.0e-12)
    dc = (u == 0) & (v == 0)
    low = (~dc) & (radius <= config.dct_low_radius)
    mid = (radius > config.dct_low_radius) & (radius <= config.dct_mid_radius)
    high = radius > config.dct_mid_radius
    return dc.float(), low.float(), mid.float(), high.float()


def _dct_table(config: FAConfig, y_values: Tuple[float, ...], uv_values: Tuple[float, ...]) -> Tensor:
    masks = _dct_band_masks(config)
    table = torch.empty(3, config.block_size, config.block_size)
    for channel, values in ((0, y_values), (1, uv_values), (2, uv_values)):
        table[channel] = sum(float(value) * mask for value, mask in zip(values, masks))
    return table


def _temporal_region(index: int, bins: int, config: FAConfig) -> int:
    if index == 0:
        return 0
    normalized = index / max(bins - 1, 1)
    if normalized <= config.temporal_low_ratio:
        return 1
    if normalized <= config.temporal_mid_ratio:
        return 2
    return 3


def _temporal_table(config: FAConfig, y_values: Tuple[float, ...], uv_values: Tuple[float, ...]) -> Tensor:
    bins = config.num_frames // 2 + 1
    table = torch.empty(3, bins)
    for index in range(bins):
        region = _temporal_region(index, bins, config)
        table[0, index] = y_values[region]
        table[1, index] = uv_values[region]
        table[2, index] = uv_values[region]
    return table


class FrequencyAnonymizationBlock(nn.Module):
    def __init__(self, config: FAConfig, block_index: int = 0):
        super().__init__()
        self.config = config
        self.block_index = block_index
        self.dct_block = config.block_size
        self.expected_frames = config.num_frames
        self.expected_freq_bins = config.num_frames // 2 + 1
        self.expected_patch_rows = math.ceil(config.image_size / config.block_size)
        self.expected_patch_cols = self.expected_patch_rows

        dct_dim = 3 * config.block_size * config.block_size
        baseline = PatchPolicyMLP(
            dct_dim, config.hidden_dim, 2 * dct_dim,
            config.mlp_depth, config.mlp_dropout,
        )
        self.dct_temporal_position = nn.Parameter(torch.zeros(1, config.num_frames, 1, 1, dct_dim))
        self.dct_row_position = nn.Parameter(torch.zeros(1, 1, self.expected_patch_rows, 1, dct_dim))
        self.dct_col_position = nn.Parameter(torch.zeros(1, 1, 1, self.expected_patch_cols, dct_dim))
        for position in (self.dct_temporal_position, self.dct_row_position, self.dct_col_position):
            nn.init.trunc_normal_(position, std=0.02)
        self.mapping_s = ResidualFrequencyAttentionPolicy(
            baseline, config.hidden_dim,
            self.expected_patch_rows, self.expected_patch_cols,
            config, config.seed + 7000 + 2 * block_index,
        )

        dct_gate_init = _dct_table(config, config.y_dct_gate_init, config.uv_dct_gate_init)
        dct_gate_init[1:3] *= config.dct_uv_gate_init_scale
        dct_sigma_init = _dct_table(config, config.y_dct_sigma_init, config.uv_dct_sigma_init)
        dct_bias = torch.cat([
            _bounded_logit(dct_gate_init, config.gate_min, config.gate_max).flatten(),
            _bounded_logit(dct_sigma_init, 0.0, config.dct_sigma_max).flatten(),
        ])
        with torch.no_grad():
            self.mapping_s.policy_head.bias.copy_(dct_bias)

        fft_gate_init = _temporal_table(config, config.y_fft_gate_init, config.uv_fft_gate_init)
        fft_sigma_init = _temporal_table(config, config.y_fft_sigma_init, config.uv_fft_sigma_init)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + 17000 + block_index)
            self.mapping_t = PatchBandFFTPolicy(
                config, self.expected_patch_rows, self.expected_patch_cols,
                fft_gate_init, fft_sigma_init,
            )

        self.register_buffer(
            "dct_privacy_profile",
            _dct_table(config, config.y_dct_guidance, config.uv_dct_guidance).clamp(0.0, 1.5),
            persistent=False,
        )
        self.register_buffer(
            "temporal_privacy_profile",
            _temporal_table(config, config.y_fft_guidance, config.uv_fft_guidance).clamp(0.0, 1.5),
            persistent=False,
        )
        self.register_buffer("_dct_matrix_cache", None, persistent=False)

        self.last_dct_gate_train: Optional[Tensor] = None
        self.last_fft_gate_train: Optional[Tensor] = None
        self.last_privacy_mask_train: Optional[Tensor] = None

    def _dct_matrix(self, x: Tensor) -> Tensor:
        cached = self._dct_matrix_cache
        if cached is None or cached.device != x.device or cached.dtype != x.dtype:
            cached = make_dct_matrix(self.dct_block, x.device, x.dtype)
            self._dct_matrix_cache = cached
        return cached

    def _privacy_mask(self, yuv: Tensor) -> Tensor:
        cfg = self.config
        y = yuv[:, 0:1].clamp(0.0, 1.0)
        cb = (yuv[:, 1:2] + 0.5).clamp(0.0, 1.0)
        cr = (yuv[:, 2:3] + 0.5).clamp(0.0, 1.0)
        skin_distance = (
            ((cb - cfg.mask_cb_center) / cfg.mask_cb_sigma).square()
            + ((cr - cfg.mask_cr_center) / cfg.mask_cr_sigma).square()
            + 0.35 * ((y - cfg.mask_luma_center) / cfg.mask_luma_sigma).square()
        )
        skin = torch.exp(-0.5 * skin_distance)
        if y.size(2) > 1:
            motion = (y[:, :, 1:] - y[:, :, :-1]).abs()
            motion = torch.cat([motion[:, :, 0:1], motion], dim=2)
            flat = motion.flatten(2)
            denominator = (
                flat.mean(dim=2).view(y.size(0), 1, 1, 1, 1)
                + 2.0 * flat.std(dim=2, unbiased=False).view(y.size(0), 1, 1, 1, 1)
                + 1.0e-6
            )
            motion = (motion / denominator).clamp(0.0, 1.0)
        else:
            motion = torch.zeros_like(y)
        mask = (
            cfg.mask_skin_weight * skin + cfg.mask_motion_weight * motion
        ) / (cfg.mask_skin_weight + cfg.mask_motion_weight)
        kernel = cfg.mask_blur_kernel if cfg.mask_blur_kernel % 2 else cfg.mask_blur_kernel + 1
        if kernel > 1:
            mask = F.avg_pool3d(mask, (1, kernel, kernel), stride=1, padding=(0, kernel // 2, kernel // 2))
        return mask.clamp(0.0, 1.0).pow(cfg.mask_power).detach()

    def _mask_per_frame_patch(self, mask: Tensor) -> Tensor:
        padded, _ = pad_to_blocks(mask, self.dct_block)
        return blockify(padded, self.dct_block).mean(dim=(5, 6)).clamp(0.0, 1.0)

    def _mask_per_video_patch(self, mask: Tensor) -> Tensor:
        padded, _ = pad_to_blocks(mask, self.dct_block)
        return blockify(padded, self.dct_block).mean(dim=(2, 5, 6)).clamp(0.0, 1.0)

    def _dct_policy(self, coeff_clip: Tensor, clip: Tensor) -> Tuple[Tensor, Tensor]:
        cfg = self.config
        batch, channels, frames, rows, cols, block, _ = coeff_clip.shape
        scale = clip.view(batch, channels, 1, 1, 1, block, block)
        normalized = (coeff_clip / (scale + 1.0e-12)).clamp(-1.0, 1.0)
        tokens = normalized.permute(0, 2, 3, 4, 1, 5, 6).contiguous().flatten(4)
        tokens = tokens + (
            self.dct_temporal_position + self.dct_row_position + self.dct_col_position
        ).to(tokens)
        use_amp = cfg.policy_bf16 and tokens.is_cuda
        with torch.amp.autocast(tokens.device.type, dtype=torch.bfloat16, enabled=use_amp):
            raw = self.mapping_s(tokens.reshape(batch * frames, rows, cols, -1))
        raw = raw.float().view(batch, frames, rows, cols, -1)
        raw_gate, raw_sigma = raw.chunk(2, dim=-1)
        raw_gate = raw_gate.view(batch, frames, rows, cols, 3, block, block).permute(0, 4, 1, 2, 3, 5, 6).contiguous()
        raw_sigma = raw_sigma.view(batch, frames, rows, cols, 3, block, block).permute(0, 4, 1, 2, 3, 5, 6).contiguous()
        gate = cfg.gate_min + (cfg.gate_max - cfg.gate_min) * torch.sigmoid(raw_gate)
        sigma = cfg.dct_sigma_max * torch.sigmoid(raw_sigma)
        return gate, sigma

    def _fft_tokens(self, spectrum: Tensor) -> Tuple[Tensor, Tensor, Tuple[int, int]]:
        batch, channels, frequencies, _, _ = spectrum.shape
        global_scale = torch.sqrt(
            spectrum.abs().float().square().mean(dim=(2, 3, 4), keepdim=True) + 1.0e-8
        ).to(spectrum.real.dtype)
        normalized = spectrum / global_scale
        real, original_hw = pad_to_blocks(normalized.real, self.dct_block)
        imag, _ = pad_to_blocks(normalized.imag, self.dct_block)
        real_blocks = blockify(real, self.dct_block)
        imag_blocks = blockify(imag, self.dct_block)
        rows, cols = real_blocks.size(3), real_blocks.size(4)
        point_tokens = torch.cat([
            real_blocks.permute(0, 3, 4, 2, 5, 6, 1),
            imag_blocks.permute(0, 3, 4, 2, 5, 6, 1),
        ], dim=-1).contiguous()
        coeff = point_tokens.flatten(4)
        log_scale = torch.log(global_scale.reshape(batch, channels) + 1.0e-8)
        scale_tokens = log_scale.view(batch, 1, 1, 1, channels).expand(
            batch, rows, cols, frequencies, channels
        )
        patch_tokens = torch.cat([coeff, scale_tokens], dim=-1)
        return patch_tokens, point_tokens, original_hw

    def _fft_policy(self, spectrum: Tensor) -> Tuple[Tensor, Tensor]:
        patch_tokens, point_tokens, original_hw = self._fft_tokens(spectrum)
        raw = self.mapping_t(patch_tokens, point_tokens)
        raw_gate, raw_sigma = raw[..., :3], raw[..., 3:]

        def restore(values: Tensor) -> Tensor:
            blocks = values.permute(0, 6, 3, 1, 2, 4, 5).contiguous()
            image = unblockify(blocks)
            height, width = original_hw
            return image[..., :height, :width].contiguous()

        raw_gate = restore(raw_gate)
        raw_sigma = restore(raw_sigma)
        gate = self.config.gate_min + (
            self.config.gate_max - self.config.gate_min
        ) * torch.sigmoid(raw_gate)
        sigma = self.config.fft_sigma_max * torch.sigmoid(raw_sigma)
        return gate, sigma

    def _apply_dct(self, yuv: Tensor, mask: Tensor, rho: float) -> Tuple[Tensor, Dict[str, Tensor]]:
        cfg = self.config
        matrix = self._dct_matrix(yuv)
        coeff, original_hw = block_dct2(yuv, self.dct_block, matrix)
        coeff_clip, clip = clip_dct_coefficients(coeff, cfg.dct_clip_quantile, cfg.clip_min)
        gate, sigma = self._dct_policy(coeff_clip, clip)

        batch, channels, frames, rows, cols, block, _ = coeff.shape
        mask_patch = self._mask_per_frame_patch(mask)
        guidance = mask_patch.view(batch, 1, frames, rows, cols, 1, 1) * self.dct_privacy_profile.view(1, 3, 1, 1, 1, block, block)
        guidance = guidance.clamp(0.0, 1.5)
        gate = (gate * (1.0 - cfg.dct_mask_gate_drop * guidance)).clamp(cfg.gate_min, cfg.gate_max)
        sigma = (sigma + cfg.dct_mask_noise_gain * guidance).clamp(0.0, cfg.dct_sigma_max)
        self.last_dct_gate_train = gate

        coeff_base = coeff + rho * (coeff_clip - coeff)
        gate_upper = 1.0 if rho == 0.0 else cfg.gate_max
        effective_gate = (1.0 - rho * (1.0 - gate)).clamp(cfg.gate_min, gate_upper)
        effective_sigma = (rho * sigma).clamp(0.0, cfg.dct_sigma_max)
        noise_scale = clip.view(batch, channels, 1, 1, 1, block, block) * effective_sigma
        noise_scale = torch.minimum(
            noise_scale,
            clip.view(batch, channels, 1, 1, 1, block, block) * cfg.dct_noise_cap_ratio,
        )
        noise = sample_laplace(coeff.shape, coeff.device, coeff.dtype) * noise_scale
        encrypted = effective_gate * coeff_base + (1.0 - effective_gate) * noise
        reconstructed = block_idct2(encrypted, original_hw, matrix)
        reconstructed = torch.cat([
            reconstructed[:, 0:1].clamp(0.0, 1.0),
            reconstructed[:, 1:3].clamp(-0.5, 0.5),
        ], dim=1)
        return reconstructed, {
            "coeff": coeff,
            "encrypted_coeff": encrypted,
            "gate": effective_gate,
            "sigma": effective_sigma,
            "mask": mask,
            "mask_patch": mask_patch,
        }

    def _apply_fft(self, yuv: Tensor, mask: Tensor, rho: float) -> Tuple[Tensor, Dict[str, Tensor]]:
        cfg = self.config
        batch, _, frames, height, width = yuv.shape
        spectrum = torch.fft.rfft(yuv, dim=2, norm=cfg.fft_norm)
        spectrum_clip, clip = clip_complex_spectrum(spectrum, cfg.fft_clip_quantile, cfg.clip_min)
        gate, sigma = self._fft_policy(spectrum)

        mask_patch = self._mask_per_video_patch(mask)
        mask_image = F.interpolate(
            mask_patch, size=(height, width), mode="bilinear", align_corners=False,
        ).unsqueeze(2)
        guidance = mask_image * self.temporal_privacy_profile.view(
            1, 3, self.expected_freq_bins, 1, 1
        )
        guidance = guidance.clamp(0.0, 1.5)
        gate = (gate * (1.0 - cfg.fft_mask_gate_drop * guidance)).clamp(
            cfg.gate_min, cfg.gate_max
        )
        sigma = (sigma + cfg.fft_mask_noise_gain * guidance).clamp(
            0.0, cfg.fft_sigma_max
        )
        self.last_fft_gate_train = gate

        spectrum_base = spectrum + rho * (spectrum_clip - spectrum)
        effective_gate = (1.0 - rho * (1.0 - gate)).clamp(cfg.gate_min, 1.0)
        effective_sigma = (rho * sigma).clamp(0.0, cfg.fft_sigma_max)
        noise_scale = torch.minimum(clip * effective_sigma, clip * cfg.fft_noise_cap_ratio)
        noise = sample_complex_laplace(spectrum, noise_scale, frames)
        encrypted = effective_gate * spectrum_base + (1.0 - effective_gate) * noise
        reconstructed = torch.fft.irfft(encrypted, n=frames, dim=2, norm=cfg.fft_norm)
        reconstructed = torch.cat([
            reconstructed[:, 0:1].clamp(0.0, 1.0),
            reconstructed[:, 1:3].clamp(-0.5, 0.5),
        ], dim=1)
        return reconstructed, {
            "spectrum": spectrum,
            "encrypted_spectrum": encrypted,
            "gate_point": effective_gate,
            "sigma_point": effective_sigma,
            "gate": effective_gate,
            "sigma": effective_sigma,
            "mask_patch": mask_patch,
        }

    def forward(self, rgb: Tensor, strength: float = 1.0, return_details: bool = False):
        cfg = self.config
        rho = float(max(0.0, min(1.0, strength)))
        expected = (3, cfg.num_frames, cfg.image_size, cfg.image_size)
        if tuple(rgb.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{expected}], received {tuple(rgb.shape)}")
        yuv = rgb_to_centered_yuv(rgb.float().clamp(0.0, 1.0))
        mask = self._privacy_mask(yuv)
        self.last_privacy_mask_train = mask
        dct_yuv, dct_details = self._apply_dct(yuv, mask, rho)
        fft_yuv, fft_details = self._apply_fft(dct_yuv, mask, rho)
        output = centered_yuv_to_rgb(fft_yuv).clamp(0.0, 1.0)
        if return_details:
            return output, {"privacy_mask": mask, "dct": dct_details, "fft": fft_details}
        return output

class SpectraSafeAnonymizer(nn.Module):
    """Paper-spec SpectraSafe anonymizer; the main experiment uses one block."""

    def __init__(self, config: Optional[FAConfig] = None):
        super().__init__()
        self.config = config or FAConfig()
        self.blocks = nn.ModuleList([
            FrequencyAnonymizationBlock(self.config, block_index=index)
            for index in range(self.config.num_blocks)
        ])
        self.num_dynamic_blocks = self.config.num_blocks
        self.learnable_budget_param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.last_records = []

    def _forward_impl(
        self,
        x: Tensor,
        strength: float,
        return_intermediates: bool,
        collect_visual: bool = False,
    ):
        rho = float(max(0.0, min(1.0, strength)))
        device_type = x.device.type if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            output = x.float()
            intermediates = []
            records = []
            for block in self.blocks:
                if collect_visual:
                    output, details = block(output, strength=rho, return_details=True)
                    records.append(details)
                else:
                    output = block(output, strength=rho)
                if return_intermediates:
                    intermediates.append(output)
            if collect_visual:
                self.last_records = records
        return (output.float(), intermediates) if return_intermediates else output.float()

    def forward(self, x: Tensor, collect_visual: bool = False, strength: float = 1.0) -> Tensor:
        return self._forward_impl(
            x,
            strength=strength,
            return_intermediates=False,
            collect_visual=collect_visual,
        )

    def forward_with_intermediates(self, x: Tensor, strength: float = 1.0):
        return self._forward_impl(x, strength=strength, return_intermediates=True)

FrequencyAnonymizer = SpectraSafeAnonymizer
