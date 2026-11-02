# -*- coding: utf-8 -*-
"""Small shared construction utilities used by all three stages."""

import random

import numpy as np
import torch

from fa_model import FAConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_fa_config(config) -> FAConfig:
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
