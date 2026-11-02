# -*- coding: utf-8 -*-
"""Thin wrappers around official VideoMAEv2-B and torchvision ResNet-50."""

import os
import sys
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def sample_frame_indices(batch: int, frames: int, count: int, device, mode: str) -> torch.Tensor:
    count = min(max(1, int(count)), int(frames))
    if mode == "fixed_uniform":
        index = torch.linspace(0, frames - 1, count, device=device).round().long()
        return index.unsqueeze(0).expand(batch, -1)
    if mode != "stratified_random":
        raise ValueError(f"Unknown privacy frame sampling mode: {mode}")
    segment = torch.arange(count, device=device)
    left = torch.div(segment * frames, count, rounding_mode="floor")
    right = torch.div((segment + 1) * frames, count, rounding_mode="floor")
    width = (right - left).clamp_min(1)
    offset = torch.floor(torch.rand(batch, count, device=device) * width).long()
    return left.unsqueeze(0) + offset


class VideoMAEv2ActionHead(nn.Module):
    """VideoMAEv2-B classifier loaded from the official local repository."""

    def __init__(self, config, pretrained: bool = True):
        super().__init__()
        if config.videomaev2_root not in sys.path:
            sys.path.insert(0, config.videomaev2_root)
        from models.modeling_finetune import vit_base_patch16_224

        self.model = vit_base_patch16_224(
            pretrained=False,
            num_classes=config.num_classes,
            all_frames=config.num_frames,
            tubelet_size=2,
            drop_path_rate=config.videomaev2_drop_path,
            head_drop_rate=config.videomaev2_head_drop_rate,
            init_scale=config.videomaev2_init_scale,
        )
        if pretrained:
            self._load_backbone(config.videomaev2_checkpoint)
        self.register_buffer(
            "internal_mean", torch.tensor(config.videomaev2_input_mean).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "internal_std", torch.tensor(config.videomaev2_input_std).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "external_mean", torch.tensor(config.external_input_mean).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "external_std", torch.tensor(config.external_input_std).view(1, 3, 1, 1, 1), persistent=False
        )

    def _load_backbone(self, checkpoint_path: str) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"VideoMAEv2 checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint
        for key in ("model", "module", "state_dict"):
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
                state = checkpoint[key]
                break
        clean = OrderedDict()
        for name, value in state.items():
            for prefix in ("_orig_mod.", "backbone.", "encoder."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            if name not in {"head.weight", "head.bias"}:
                clean[name] = value
        missing, unexpected = self.model.load_state_dict(clean, strict=False)
        missing = [name for name in missing if not name.startswith("head.")]
        print(f"[VideoMAEv2] missing_non_head={len(missing)}, unexpected={len(unexpected)}", flush=True)

    def forward(self, externally_normalized_video: torch.Tensor) -> torch.Tensor:
        raw = externally_normalized_video.float() * self.external_std + self.external_mean
        normalized = (raw.clamp(0.0, 1.0) - self.internal_mean) / self.internal_std
        return self.model(normalized)


class VideoPrivacyHead(nn.Module):
    def __init__(self, config, pretrained: bool = True):
        super().__init__()
        weights = (
            ResNet50_Weights.IMAGENET1K_V2
            if pretrained and config.privacy_pretrained
            else None
        )
        # Initialization is a weight-import stage, so a requested pretrained
        # weight failure must be visible rather than silently falling back to
        # random parameters.  Training/evaluation pass pretrained=False when
        # a checkpoint will immediately provide every parameter.
        self.backbone = resnet50(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, config.num_privacy_attributes)
        self.frames_per_video = config.privacy_frames_per_video
        self.train_sampling = config.privacy_frame_sampling
        self.last_frame_indices: Optional[torch.Tensor] = None

    def forward(self, video: torch.Tensor, frame_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, channels, frames, height, width = video.shape
        if frame_indices is None:
            mode = self.train_sampling if self.training else "fixed_uniform"
            frame_indices = sample_frame_indices(
                batch, frames, self.frames_per_video, video.device, mode
            )
        else:
            frame_indices = frame_indices.to(video.device, dtype=torch.long)
            if frame_indices.ndim == 1:
                frame_indices = frame_indices.unsqueeze(0)
            if frame_indices.size(0) == 1 and batch > 1:
                frame_indices = frame_indices.expand(batch, -1)
        count = frame_indices.size(1)
        gather = frame_indices[:, None, :, None, None].expand(batch, channels, count, height, width)
        selected = torch.gather(video, 2, gather)
        selected = selected.permute(0, 2, 1, 3, 4).reshape(batch * count, channels, height, width)
        self.last_frame_indices = frame_indices.detach()
        return self.backbone(selected).view(batch, count, -1).mean(dim=1)


def build_heads(config, pretrained: bool = True):
    """Build evaluator architectures, optionally importing pretrained weights."""
    return (
        VideoMAEv2ActionHead(config, pretrained=pretrained),
        VideoPrivacyHead(config, pretrained=pretrained),
    )
