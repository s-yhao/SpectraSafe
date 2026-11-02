# -*- coding: utf-8 -*-
"""Clean 30-epoch adversarial + 70-epoch SpectraSafe main experiment."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from torchvision.utils import save_image

from config import (
    build_fa_config,
    config_dict,
    load_config,
    rho_for_epoch,
    seed_everything,
)
from data import make_loader
from fa_model import SpectraSafeAnonymizer
from heads import build_heads, sample_frame_indices


def set_trainable(model: nn.Module, enabled: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = enabled


def normalize_external(video: torch.Tensor, config) -> torch.Tensor:
    mean = video.new_tensor(config.external_input_mean).view(1, 3, 1, 1, 1)
    std = video.new_tensor(config.external_input_std).view(1, 3, 1, 1, 1)
    return (video - mean) / std


def action_logits(model: nn.Module, video: torch.Tensor) -> torch.Tensor:
    output = model(video)
    return output[0] if isinstance(output, (tuple, list)) else output


def make_adamw(parameters, lr: float, weight_decay: float, config):
    return torch.optim.AdamW(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
        fused=bool(config.fused_adamw and torch.cuda.is_available()),
    )


def privacy_metrics(target: np.ndarray, probability: np.ndarray, threshold: float):
    class_ap = []
    for index in range(target.shape[1]):
        truth = target[:, index]
        if len(truth) == 0 or np.all(truth == truth[0]):
            class_ap.append(float("nan"))
        else:
            class_ap.append(float(average_precision_score(truth, probability[:, index])))
    valid = [value for value in class_ap if np.isfinite(value)]
    cmap = float(np.mean(valid)) if valid else float("nan")
    prediction = (probability >= threshold).astype(np.int64)
    f1 = float(f1_score(target, prediction, average="macro", zero_division=0)) if len(target) else 0.0
    return cmap, f1, class_ap


def save_preview(epoch_name: str, frames: torch.Tensor, fa: nn.Module, directory: Path, device) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        x = frames[:1].permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        rho = 1.0 if epoch_name.startswith("benchmark") else rho_for_epoch(int(epoch_name.split("_")[-1]))
        y = fa(x, strength=rho).clamp(0.0, 1.0)
        count = min(8, x.size(2))
        index = torch.linspace(0, x.size(2) - 1, count, device=device).round().long()
        original = x[0, :, index].permute(1, 0, 2, 3)
        anonymous = y[0, :, index].permute(1, 0, 2, 3)
        difference = (original - anonymous).abs().mul(3.0).clamp(0.0, 1.0)
        save_image(
            torch.cat([original, anonymous, difference]).cpu(),
            directory / f"{epoch_name}.png",
            nrow=count,
            padding=4,
        )


def adversarial_epoch(epoch, loader, fa, ft, fb, optimizers, criteria, scaler, device, config):
    fa.train(); ft.train(); fb.train()
    rho = rho_for_epoch(epoch)
    use_amp = bool(config.amp and device.type == "cuda")
    update_fa = epoch > 1
    totals = {"fa": 0.0, "fa_action": 0.0, "fa_privacy": 0.0, "ft": 0.0, "fb": 0.0}
    batches = 0

    for batch_index, (frames, action_target, privacy_target, _) in enumerate(loader):
        batches += 1
        x = frames.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        action_target = action_target.to(device, non_blocking=True)
        privacy_target = privacy_target.to(device, non_blocking=True)
        sampled = sample_frame_indices(
            x.size(0), x.size(2), config.privacy_frames_per_video,
            device, config.privacy_frame_sampling,
        )

        if update_fa:
            set_trainable(fa, True); set_trainable(ft, False); set_trainable(fb, False)
            optimizers["fa"].zero_grad(set_to_none=True)
            anonymous, _ = fa.forward_with_intermediates(x, strength=rho)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                normalized = normalize_external(anonymous, config)
                fa_action = criteria["action"](action_logits(ft, normalized), action_target)
                fa_privacy = criteria["privacy"](fb(normalized, frame_indices=sampled), privacy_target)
                fa_loss = (
                    config.action_loss_weight * fa_action
                    - config.privacy_adversarial_weight * fa_privacy
                )
            scaler.scale(fa_loss).backward()
            scaler.step(optimizers["fa"])
            scaler.update()
            anonymous_detached = anonymous.detach()
            totals["fa"] += float(fa_loss.detach())
            totals["fa_action"] += float(fa_action.detach())
            totals["fa_privacy"] += float(fa_privacy.detach())
        else:
            with torch.no_grad():
                anonymous_detached = fa(x, strength=rho)

        set_trainable(fa, False); set_trainable(ft, True); set_trainable(fb, True)
        optimizers["ft"].zero_grad(set_to_none=True)
        optimizers["fb"].zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            normalized = normalize_external(anonymous_detached, config)
            ft_loss = criteria["action"](action_logits(ft, normalized), action_target)
            fb_loss = criteria["privacy"](fb(normalized, frame_indices=sampled), privacy_target)
            head_loss = config.action_loss_weight * ft_loss + config.privacy_head_loss_weight * fb_loss
        scaler.scale(head_loss).backward()
        scaler.unscale_(optimizers["ft"]); scaler.unscale_(optimizers["fb"])
        torch.nn.utils.clip_grad_norm_(ft.parameters(), config.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(fb.parameters(), config.grad_clip_norm)
        scaler.step(optimizers["ft"]); scaler.step(optimizers["fb"]); scaler.update()
        totals["ft"] += float(ft_loss.detach()); totals["fb"] += float(fb_loss.detach())

        if batch_index % config.log_interval == 0:
            divisor = max(1, batches)
            print(
                f"[ADV E{epoch:03d} I{batch_index:04d}] rho={rho:.3f} "
                f"fa={totals['fa']/divisor:.4f} ft={totals['ft']/divisor:.4f} "
                f"fb={totals['fb']/divisor:.4f}", flush=True,
            )
    return {key: value / max(1, batches) for key, value in totals.items()}


def benchmark_epoch(epoch, loader, fa, ft, fb, optimizers, criteria, scaler, device, config):
    fa.eval(); ft.train(); fb.train()
    set_trainable(fa, False); set_trainable(ft, True); set_trainable(fb, True)
    use_amp = bool(config.amp and device.type == "cuda")
    totals = {"action": 0.0, "privacy": 0.0, "accuracy": 0.0}
    batches = 0
    for batch_index, (frames, action_target, privacy_target, _) in enumerate(loader):
        batches += 1
        x = frames.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        action_target = action_target.to(device, non_blocking=True)
        privacy_target = privacy_target.to(device, non_blocking=True)
        with torch.no_grad():
            anonymous = fa(x, strength=1.0)
        optimizers["ft"].zero_grad(set_to_none=True); optimizers["fb"].zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            ft_logits = action_logits(ft, normalize_external(anonymous, config)).clamp(
                -config.logit_clip, config.logit_clip
            )
            fb_logits = fb(anonymous).clamp(-config.logit_clip, config.logit_clip)
            action_loss = criteria["action"](ft_logits, action_target)
            privacy_loss = criteria["privacy"](fb_logits, privacy_target)
            loss = config.action_loss_weight * action_loss + config.privacy_head_loss_weight * privacy_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizers["ft"]); scaler.unscale_(optimizers["fb"])
        torch.nn.utils.clip_grad_norm_(ft.parameters(), config.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(fb.parameters(), config.grad_clip_norm)
        scaler.step(optimizers["ft"]); scaler.step(optimizers["fb"]); scaler.update()
        totals["action"] += float(action_loss.detach())
        totals["privacy"] += float(privacy_loss.detach())
        totals["accuracy"] += float((ft_logits.argmax(1) == action_target).float().mean())
        if batch_index % config.log_interval == 0:
            print(f"[BENCH E{epoch:03d} I{batch_index:04d}]", flush=True)
    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def validate(loader, fa, ft, fb, criteria, device, config, benchmark: bool, rho: float):
    fa.eval(); ft.eval(); fb.eval()
    use_amp = bool(config.amp and device.type == "cuda")
    total = correct = 0
    action_loss = privacy_loss = 0.0
    targets, probabilities = [], []
    for frames, action_target, privacy_target, _ in loader:
        x = frames.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        action_target = action_target.to(device, non_blocking=True)
        privacy_target = privacy_target.to(device, non_blocking=True)
        anonymous = fa(x, strength=rho)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            ft_logits = action_logits(ft, normalize_external(anonymous, config))
            privacy_input = anonymous if benchmark else normalize_external(anonymous, config)
            fb_logits = fb(privacy_input)
            if benchmark:
                ft_logits = ft_logits.clamp(-config.logit_clip, config.logit_clip)
                fb_logits = fb_logits.clamp(-config.logit_clip, config.logit_clip)
            action_batch_loss = criteria["action"](ft_logits, action_target)
            privacy_batch_loss = criteria["privacy"](fb_logits, privacy_target)
        batch = x.size(0)
        total += batch
        correct += int((ft_logits.argmax(1) == action_target).sum())
        action_loss += float(action_batch_loss) * batch
        privacy_loss += float(privacy_batch_loss) * batch
        targets.append(privacy_target.cpu().numpy())
        probabilities.append(torch.sigmoid(fb_logits).cpu().numpy())
    target = np.concatenate(targets) if targets else np.zeros((0, 5), dtype=np.float32)
    probability = np.concatenate(probabilities) if probabilities else np.zeros_like(target)
    cmap, f1, class_ap = privacy_metrics(target, probability, config.privacy_threshold)
    return {
        "action_acc": correct / max(1, total),
        "action_loss": action_loss / max(1, total),
        "privacy_loss": privacy_loss / max(1, total),
        "cmap": cmap,
        "f1": f1,
        "class_ap": class_ap,
    }


def checkpoint_state(epoch, phase, fa, ft, fb, optimizers, benchmark_heads_fresh):
    return {
        "epoch": epoch + 1,
        "phase": phase,
        "fa_model_state_dict": fa.state_dict(),
        "ft_model_state_dict": ft.state_dict(),
        "fb_model_state_dict": fb.state_dict(),
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "benchmark_heads_fresh": benchmark_heads_fresh,
    }


def save_checkpoint(path, *args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_state(*args), path)


def load_clean_checkpoint(path, fa, ft, fb, optimizers, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    fa.load_state_dict(checkpoint["fa_model_state_dict"], strict=True)
    ft.load_state_dict(checkpoint["ft_model_state_dict"], strict=False)
    fb.load_state_dict(checkpoint["fb_model_state_dict"], strict=False)
    for name, state in checkpoint.get("optimizers", {}).items():
        if name in optimizers:
            optimizers[name].load_state_dict(state)
            for value in optimizers[name].state.values():
                for key, tensor in value.items():
                    if torch.is_tensor(tensor):
                        value[key] = tensor.to(device)
    return int(checkpoint["epoch"]), bool(checkpoint.get("benchmark_heads_fresh", False))


def load_initialization_checkpoint(path, config, fa, ft, fb) -> None:
    """Load the stage-1 weight-import checkpoint; no optimizer state exists."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("phase") != "initialization":
        raise ValueError(f"Not an initialization checkpoint: {path}")
    if checkpoint.get("trained", True):
        raise ValueError("Initialization checkpoint unexpectedly reports trained=True")
    saved_config = checkpoint.get("config", {})
    for name in (
        "dataset", "num_classes", "num_privacy_attributes", "num_frames",
        "image_size", "anonymizer_blocks", "mask_skin_weight",
        "mask_motion_weight", "mask_power", "fft_point_hidden_dim",
    ):
        if name in saved_config and saved_config[name] != getattr(config, name):
            raise ValueError(
                f"Initialization/config mismatch for {name}: "
                f"{saved_config[name]!r} vs {getattr(config, name)!r}"
            )
    if fa is not None:
        fa.load_state_dict(checkpoint["fa_model_state_dict"], strict=True)
    ft.load_state_dict(checkpoint["ft_model_state_dict"], strict=True)
    fb.load_state_dict(checkpoint["fb_model_state_dict"], strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config")
    parser.add_argument("--output-dir", default=None, help="Optional output override")
    parser.add_argument(
        "--initialization-checkpoint",
        default=None,
        help="Defaults to <output_dir>/checkpoints/initialization.pth",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--adversarial-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, require_pretrained_checkpoint=False)
    if args.output_dir is not None:
        config.output_dir = str(Path(args.output_dir).expanduser().resolve())
    if args.no_resume:
        config.resume = False

    seed_everything(config.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(config.output_dir)
    checkpoints = output / "checkpoints"
    visualizations = output / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "resolved_config.json", "w", encoding="utf-8") as handle:
        json.dump(config_dict(config), handle, ensure_ascii=False, indent=2)

    temporary = checkpoints / "model_temp.pth"
    resume_available = bool(config.resume and temporary.is_file())
    initialization_checkpoint = (
        Path(args.initialization_checkpoint).expanduser().resolve()
        if args.initialization_checkpoint
        else checkpoints / "initialization.pth"
    )
    if not initialization_checkpoint.is_file():
        raise FileNotFoundError(
            "Initialization checkpoint not found. Run "
            "`python -m initialization.initialize --config <config>` first, "
            f"or pass --initialization-checkpoint. Expected: {initialization_checkpoint}"
        )

    fa = SpectraSafeAnonymizer(build_fa_config(config)).to(device)
    # Initialization or resume checkpoints provide the initial head weights;
    # avoid importing them a second time here.
    ft, fb = build_heads(config, pretrained=False); ft = ft.to(device); fb = fb.to(device)
    if not resume_available:
        load_initialization_checkpoint(initialization_checkpoint, config, fa, ft, fb)
        print(f"[Initialization] loaded {initialization_checkpoint}", flush=True)
    criteria = {
        "action": nn.CrossEntropyLoss().to(device),
        "privacy": nn.BCEWithLogitsLoss(
            weight=torch.tensor(config.privacy_attribute_weights, device=device)
        ),
    }
    optimizers = {
        "fa": make_adamw(fa.parameters(), config.lr_fa, config.fa_weight_decay, config),
        "ft": make_adamw(ft.parameters(), config.lr_ft, config.head_weight_decay, config),
        "fb": make_adamw(fb.parameters(), config.lr_fb, config.head_weight_decay, config),
    }
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.amp and device.type == "cuda"))
    start_epoch, benchmark_heads_fresh = 1, False
    if resume_available:
        start_epoch, benchmark_heads_fresh = load_clean_checkpoint(
            temporary, fa, ft, fb, optimizers, device
        )
        print(f"[Resume] next global epoch={start_epoch}", flush=True)

    adversarial_train = make_loader(config, "train", benchmark=False)
    adversarial_val = make_loader(config, "val", benchmark=False)
    benchmark_train = make_loader(config, "train", benchmark=True)
    benchmark_val = make_loader(config, "val", benchmark=True)

    total_epochs = config.adversarial_epochs + (0 if args.adversarial_only else config.benchmark_epochs)
    for global_epoch in range(start_epoch, total_epochs + 1):
        started = time.time()
        if global_epoch <= config.adversarial_epochs:
            phase = "adversarial"
            metrics = adversarial_epoch(
                global_epoch, adversarial_train, fa, ft, fb,
                optimizers, criteria, scaler, device, config,
            )
            if global_epoch == 1 or global_epoch % config.adversarial_val_every == 0:
                validation = validate(
                    adversarial_val, fa, ft, fb, criteria, device, config,
                    benchmark=False, rho=rho_for_epoch(global_epoch),
                )
                print(f"[VAL ADV E{global_epoch:03d}] {validation}", flush=True)
            preview_frames = next(iter(adversarial_val))[0]
            save_preview(f"adversarial_{global_epoch}", preview_frames, fa, visualizations, device)
        else:
            phase = "benchmark"
            benchmark_epoch_number = global_epoch - config.adversarial_epochs
            if not benchmark_heads_fresh:
                del ft, fb
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                ft, fb = build_heads(config, pretrained=False)
                load_initialization_checkpoint(
                    initialization_checkpoint, config, None, ft, fb
                )
                ft, fb = ft.to(device), fb.to(device)
                optimizers = {
                    "ft": make_adamw(ft.parameters(), config.benchmark_lr_ft, config.benchmark_weight_decay, config),
                    "fb": make_adamw(fb.parameters(), config.benchmark_lr_fb, config.benchmark_weight_decay, config),
                }
                benchmark_heads_fresh = True
            metrics = benchmark_epoch(
                benchmark_epoch_number, benchmark_train, fa, ft, fb,
                optimizers, criteria, scaler, device, config,
            )
            if benchmark_epoch_number % config.benchmark_val_every == 0:
                validation = validate(
                    benchmark_val, fa, ft, fb, criteria, device, config,
                    benchmark=True, rho=1.0,
                )
                print(
                    f"[VAL BENCH E{benchmark_epoch_number:03d} "
                    f"GLOBAL_E{global_epoch:03d}] {validation}",
                    flush=True,
                )
                preview_frames = next(iter(benchmark_val))[0]
                save_preview(f"benchmark_{benchmark_epoch_number}", preview_frames, fa, visualizations, device)

        save_checkpoint(
            temporary, global_epoch, phase, fa, ft, fb, optimizers, benchmark_heads_fresh
        )
        local_epoch = global_epoch if phase == "adversarial" else global_epoch - config.adversarial_epochs
        if local_epoch % config.save_every == 0 or global_epoch == total_epochs:
            save_checkpoint(
                checkpoints / f"{phase}_epoch{local_epoch:03d}.pth",
                global_epoch, phase, fa, ft, fb, optimizers, benchmark_heads_fresh,
            )
        print(f"[{phase}] epoch={local_epoch} metrics={metrics} time={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
