# -*- coding: utf-8 -*-
"""Create the SpectraSafe initialization checkpoint without training.

This stage only constructs the anonymizer and imports the configured
VideoMAE V2-B and ResNet-50 pretrained weights.  It does not optimize an
identity/reconstruction objective and does not read the experiment dataset.

Run from the repository root:

    python -m initialization.initialize --config configs/vpucf.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import build_fa_config, config_dict, load_config, seed_everything
from fa_model import SpectraSafeAnonymizer
from heads import build_heads


def initialization_state(config, fa, ft, fb) -> dict:
    return {
        "format_version": 1,
        "phase": "initialization",
        "epoch": 0,
        "trained": False,
        "initialization_method": "pretrained_weight_import_only",
        "fa_model_state_dict": fa.state_dict(),
        "ft_model_state_dict": ft.state_dict(),
        "fb_model_state_dict": fb.state_dict(),
        "config": config_dict(config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config")
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to <output_dir>/checkpoints/initialization.pth",
    )
    args = parser.parse_args()

    # Dataset path variables are intentionally not required in this stage.
    config = load_config(args.config, require_dataset_paths=False)
    seed_everything(config.seed)
    # Building the heads imports the configured Kinetics-710 VideoMAE V2-B
    # checkpoint and torchvision ImageNet-1K V2 ResNet-50 weights.  No dataset,
    # optimizer, loss, backward pass, or training loop exists in this stage.
    # Keep all tensors on CPU so initialization does not require or reserve a
    # GPU and produces a portable checkpoint.
    fa = SpectraSafeAnonymizer(build_fa_config(config))
    ft, fb = build_heads(config)

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(config.output_dir) / "checkpoints" / "initialization.pth"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(initialization_state(config, fa, ft, fb), output)

    manifest = {
        "checkpoint": str(output),
        "phase": "initialization",
        "trained": False,
        "method": "pretrained_weight_import_only",
        "videomaev2_checkpoint": config.videomaev2_checkpoint,
        "privacy_backbone": "torchvision ResNet-50 IMAGENET1K_V2",
        "seed": config.seed,
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
