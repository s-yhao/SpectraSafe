# -*- coding: utf-8 -*-
"""Evaluate a clean SpectraSafe checkpoint on VP-UCF or VP-HMDB."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn

from config import load_config
from data import make_loader
from fa_model import FrequencyAnonymizer
from heads import build_heads
from runtime import build_fa_config, seed_everything
from training.train import validate


def load_checkpoint(path: str, fa, ft, fb) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"fa_model_state_dict", "ft_model_state_dict", "fb_model_state_dict"}
    missing = required - set(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint is missing keys: {sorted(missing)}")
    fa.load_state_dict(checkpoint["fa_model_state_dict"], strict=True)
    ft.load_state_dict(checkpoint["ft_model_state_dict"], strict=True)
    fb.load_state_dict(checkpoint["fb_model_state_dict"], strict=True)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--phase",
        choices=("adversarial", "benchmark"),
        default="benchmark",
        help="Benchmark uses raw [0,1] frames for the privacy evaluator.",
    )
    parser.add_argument("--rho", type=float, default=1.0)
    args = parser.parse_args()

    config = load_config(args.config, require_pretrained_checkpoint=False)
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_loader(config, "val", benchmark=args.phase == "benchmark")
    fa = FrequencyAnonymizer(build_fa_config(config)).to(device)
    # The evaluation checkpoint supplies all trained weights.
    ft, fb = build_heads(config, pretrained=False)
    ft, fb = ft.to(device), fb.to(device)
    checkpoint = load_checkpoint(args.checkpoint, fa, ft, fb)
    criteria = {
        "action": nn.CrossEntropyLoss().to(device),
        "privacy": nn.BCEWithLogitsLoss(
            weight=torch.tensor(config.privacy_attribute_weights, device=device)
        ),
    }
    metrics = validate(
        loader,
        fa,
        ft,
        fb,
        criteria,
        device,
        config,
        benchmark=args.phase == "benchmark",
        rho=float(args.rho),
    )
    result = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_phase": checkpoint.get("phase"),
        "evaluation_phase": args.phase,
        "rho": float(args.rho),
        **metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
