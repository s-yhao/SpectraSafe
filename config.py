# -*- coding: utf-8 -*-
"""Configuration loading for the clean SpectraSafe implementation."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List

import numpy as np
import torch

from fa_model import FAConfig


@dataclass
class ExperimentConfig:
    # Experiment
    experiment_name: str = "spectrasafe"
    output_dir: str = "outputs/spectrasafe"

    # Dataset. Paths are deliberately external to the repository.
    dataset: str = "vpucf"
    video_root: str = ""
    split_root: str = ""
    privacy_label_pkl: str = ""
    action_classes_file: str = ""
    split_index: int = 1
    num_classes: int = 101
    num_privacy_attributes: int = 5
    num_frames: int = 16
    image_size: int = 224
    data_percentage: float = 1.0

    # Official VideoMAE V2 repository and Kinetics-710 distilled checkpoint.
    videomaev2_root: str = ""
    videomaev2_checkpoint: str = ""
    videomaev2_drop_path: float = 0.1
    videomaev2_head_drop_rate: float = 0.0
    videomaev2_init_scale: float = 0.001
    videomaev2_input_mean: List[float] = field(
        default_factory=lambda: [0.5, 0.5, 0.5]
    )
    videomaev2_input_std: List[float] = field(
        default_factory=lambda: [0.5, 0.5, 0.5]
    )
    external_input_mean: List[float] = field(
        default_factory=lambda: [0.45, 0.45, 0.45]
    )
    external_input_std: List[float] = field(
        default_factory=lambda: [0.225, 0.225, 0.225]
    )

    # SpectraSafe architecture from the paper.
    anonymizer_blocks: int = 1
    mask_skin_weight: float = 1.0
    mask_motion_weight: float = 1.0
    mask_power: float = 0.6
    policy_bf16: bool = True
    fft_point_hidden_dim: int = 64

    # Privacy classifier and metrics.
    privacy_frames_per_video: int = 4
    privacy_frame_sampling: str = "stratified_random"
    privacy_pretrained: bool = True
    privacy_threshold: float = 0.5
    privacy_attribute_weights: List[float] = field(
        default_factory=lambda: [1.0] * 5
    )

    # Phase 1: adversarial anonymizer training.
    adversarial_epochs: int = 30
    adversarial_batch_size: int = 8
    adversarial_val_batch_size: int = 32
    lr_fa: float = 7.0e-5
    lr_ft: float = 3.0e-5
    lr_fb: float = 5.0e-5
    fa_weight_decay: float = 1.0e-5
    head_weight_decay: float = 1.0e-4
    action_loss_weight: float = 1.0
    privacy_adversarial_weight: float = 1.0
    privacy_head_loss_weight: float = 1.15

    # Phase 2: freeze FA, reinitialize both evaluators, train for 70 epochs.
    benchmark_epochs: int = 70
    benchmark_batch_size: int = 16
    benchmark_val_batch_size: int = 16
    benchmark_lr_ft: float = 3.0e-5
    benchmark_lr_fb: float = 5.0e-5
    benchmark_weight_decay: float = 1.0e-4
    logit_clip: float = 30.0

    # Runtime.
    seed: int = 42
    num_workers: int = 12
    benchmark_num_workers: int = 8
    prefetch_factor: int = 4
    benchmark_prefetch_factor: int = 1
    amp: bool = True
    fused_adamw: bool = True
    grad_clip_norm: float = 5.0
    log_interval: int = 25
    adversarial_val_every: int = 10
    benchmark_val_every: int = 10
    save_every: int = 10
    resume: bool = True


def _expand_path(value: str, base: Path) -> str:
    if not value:
        return value
    expanded_text = os.path.expandvars(os.path.expanduser(value))
    if "${" in expanded_text:
        raise ValueError(f"Required environment variable is not set: {value}")
    expanded = Path(expanded_text)
    if not expanded.is_absolute():
        expanded = base / expanded
    return str(expanded.resolve())


def validate_config(config: ExperimentConfig) -> None:
    if config.dataset.lower() not in {"vpucf", "vphmdb"}:
        raise ValueError("dataset must be 'vpucf' or 'vphmdb'")
    if config.num_frames != 16:
        raise ValueError("The paper configuration requires num_frames=16")
    if config.image_size != 224:
        raise ValueError("The paper configuration requires image_size=224")
    if config.anonymizer_blocks != 1:
        raise ValueError("The paper main experiment uses one anonymizer block")
    if config.mask_skin_weight != 1.0 or config.mask_motion_weight != 1.0:
        raise ValueError("The paper uses equal skin/motion guidance weights")
    if config.mask_power != 0.6:
        raise ValueError("The paper main experiment uses gamma=0.6")
    if config.fft_point_hidden_dim <= 0:
        raise ValueError("fft_point_hidden_dim must be positive")
    if config.adversarial_epochs != 30 or config.benchmark_epochs != 70:
        raise ValueError("The paper protocol is 30 adversarial + 70 benchmark epochs")
    if len(config.privacy_attribute_weights) != config.num_privacy_attributes:
        raise ValueError("privacy_attribute_weights length does not match attributes")
    if not 0.0 < config.data_percentage <= 1.0:
        raise ValueError("data_percentage must be in (0, 1]")


def load_config(
    path: str,
    require_dataset_paths: bool = True,
    require_pretrained_checkpoint: bool = True,
) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    valid_names = {item.name for item in fields(ExperimentConfig)}
    unknown = sorted(set(payload) - valid_names)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {unknown}")
    config = ExperimentConfig(**payload)
    path_names = ["output_dir", "videomaev2_root"]
    if require_pretrained_checkpoint:
        path_names.append("videomaev2_checkpoint")
    if require_dataset_paths:
        path_names.extend([
            "video_root", "split_root", "privacy_label_pkl", "action_classes_file"
        ])
    for name in path_names:
        setattr(config, name, _expand_path(getattr(config, name), config_path.parent))
    validate_config(config)
    return config


def config_dict(config: ExperimentConfig) -> dict:
    return asdict(config)


def rho_for_epoch(epoch: int) -> float:
    """Paper schedule: E1=0, E2-E10 -> 0.4, E11-E30 -> 1.0."""
    if epoch <= 1:
        return 0.0
    if epoch <= 10:
        return 0.4 * float(epoch - 1) / 9.0
    return min(1.0, 0.4 + 0.6 * float(epoch - 10) / 20.0)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_fa_config(config: ExperimentConfig) -> FAConfig:
    return FAConfig(
        num_frames=config.num_frames,
        image_size=config.image_size,
        num_blocks=config.anonymizer_blocks,
        seed=config.seed,
        mask_skin_weight=config.mask_skin_weight,
        mask_motion_weight=config.mask_motion_weight,
        mask_power=config.mask_power,
        policy_bf16=config.policy_bf16,
        fft_point_hidden_dim=config.fft_point_hidden_dim,
    )
