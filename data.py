# -*- coding: utf-8 -*-
"""VP-UCF and VP-HMDB datasets for the clean SpectraSafe pipeline."""

import json
import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import decord
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


decord.bridge.set_bridge("torch")
PRIVACY_ATTRIBUTE_NAMES = [
    "face", "skin_color", "gender", "nudity", "familiar_relationship"
]


def _normalize_path(path: str) -> str:
    path = path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def _privacy_vector(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, dict):
        for nested in ("labels", "privacy", "target"):
            if nested in value:
                return _privacy_vector(value[nested])
        lowered = {str(key).lower().replace(" ", "_"): key for key in value}
        aliases = [
            PRIVACY_ATTRIBUTE_NAMES,
            ["face", "skin", "gender", "nudity", "relationship"],
            ["face", "skin_color", "gender", "nudity", "familiarity"],
        ]
        for names in aliases:
            if all(name in lowered for name in names):
                return np.asarray([value[lowered[name]] for name in names], dtype=np.float32)
    if isinstance(value, (list, tuple)) and len(value) >= 5:
        return np.asarray(value[:5], dtype=np.float32)
    if isinstance(value, np.ndarray) and value.size >= 5:
        return value.astype(np.float32).reshape(-1)[:5]
    if torch.is_tensor(value) and value.numel() >= 5:
        return value.detach().cpu().float().reshape(-1).numpy()[:5]
    return None


def _privacy_lookup(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        for key in ("labels", "data", "annos", "annotations", "privacy_labels", "video_labels"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise RuntimeError("privacy_label_pkl must contain a dictionary")
    lookup: Dict[str, np.ndarray] = {}
    for key, value in payload.items():
        vector = _privacy_vector(value)
        if vector is None:
            continue
        normalized = _normalize_path(str(key))
        basename = os.path.basename(normalized)
        stem = os.path.splitext(basename)[0]
        lookup[normalized] = vector
        lookup[basename] = vector
        lookup[stem] = vector
    return lookup


def _split_entries(path: str, is_train: bool) -> List[Tuple[str, Optional[int]]]:
    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if not parts:
                continue
            relative = _normalize_path(parts[0])
            label = int(parts[1]) - 1 if is_train and len(parts) >= 2 else None
            entries.append((relative, label))
    return entries


def _class_mapping(action_classes_file: str, train_entries) -> Dict[str, int]:
    mapping = {
        _normalize_path(path).split("/")[0]: int(label)
        for path, label in train_entries if label is not None
    }
    if not action_classes_file or not os.path.isfile(action_classes_file):
        return mapping
    with open(action_classes_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and isinstance(data.get("classes"), dict):
        data = data["classes"]
    if not isinstance(data, dict) or not data:
        return mapping
    first_key = next(iter(data))
    if str(first_key).isdigit():
        mapping.update({str(value): int(key) for key, value in data.items()})
    else:
        mapping.update({str(key): int(value) - 1 for key, value in data.items()})
    return mapping


class VPUCFDataset(Dataset):
    def __init__(
        self,
        root: str,
        splits_root: str,
        privacy_label_pkl: str,
        action_classes_file: str,
        split: str,
        split_index: int,
        num_frames: int,
        image_size: int,
        training: bool,
        data_percentage: float = 1.0,
        seed: int = 42,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.num_frames = int(num_frames)
        self.training = bool(training)
        self.rng = random.Random(seed)
        train_file = os.path.join(splits_root, f"trainlist{split_index:02d}.txt")
        test_file = os.path.join(splits_root, f"testlist{split_index:02d}.txt")
        if not os.path.isfile(train_file) or not os.path.isfile(test_file):
            raise FileNotFoundError(f"Missing UCF split files under {splits_root}")

        train_entries = _split_entries(train_file, is_train=True)
        classes = _class_mapping(action_classes_file, train_entries)
        raw_entries = train_entries if split == "train" else _split_entries(test_file, is_train=False)
        privacy = _privacy_lookup(privacy_label_pkl)
        samples = []
        for relative, label in raw_entries:
            absolute = os.path.join(root, relative)
            if not os.path.isfile(absolute):
                continue
            if label is None:
                class_name = _normalize_path(relative).split("/")[0]
                if class_name not in classes:
                    continue
                label = classes[class_name]
            basename = os.path.basename(relative)
            stem = os.path.splitext(basename)[0]
            vector = None
            for candidate in (
                _normalize_path(relative),
                _normalize_path(relative).replace(".avi", ""),
                basename,
                stem,
                _normalize_path(absolute),
            ):
                if candidate in privacy:
                    vector = privacy[candidate]
                    break
            if vector is not None:
                samples.append((absolute, int(label), vector.astype(np.float32)))
        if not samples:
            raise RuntimeError("VP-UCF is empty; check dataset and label paths")
        if 0.0 < data_percentage < 1.0:
            self.rng.shuffle(samples)
            samples = samples[:max(1, round(len(samples) * data_percentage))]
        self.samples = samples
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _indices(self, frame_count: int) -> np.ndarray:
        if frame_count <= 0:
            raise RuntimeError("Video has no frames")
        if frame_count < self.num_frames:
            return np.linspace(0, frame_count - 1, self.num_frames).round().astype(np.int64)
        if not self.training:
            return np.linspace(0, frame_count - 1, self.num_frames).astype(np.int64)
        bins = np.linspace(0, frame_count, self.num_frames + 1).astype(int)
        return np.asarray([
            self.rng.randint(bins[index], max(bins[index] + 1, bins[index + 1]) - 1)
            for index in range(self.num_frames)
        ], dtype=np.int64)

    def __getitem__(self, index: int):
        video_path, action_label, privacy_label = self.samples[index]
        try:
            reader = decord.VideoReader(video_path, ctx=decord.cpu())
            frames = reader.get_batch(self._indices(len(reader)))
            frames = torch.stack([
                self.transform(frames[i].numpy()) for i in range(frames.shape[0])
            ])
        except Exception as exc:
            raise RuntimeError(f"Failed to read {video_path}: {exc}") from exc
        return (
            frames,
            torch.tensor(action_label, dtype=torch.long),
            torch.tensor(privacy_label, dtype=torch.float32),
            video_path,
        )


def _hmdb_class_mapping(path: str) -> Dict[str, int]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Missing HMDB classInd.txt: {path}")
    mapping: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) >= 2:
                mapping[parts[1]] = int(parts[0]) - 1
    if not mapping:
        raise RuntimeError(f"HMDB class mapping is empty: {path}")
    return mapping


def _hmdb_split_entries(
    split_root: str,
    split_index: int,
    split: str,
    class_to_index: Dict[str, int],
) -> List[Tuple[str, int]]:
    target_flag = 1 if split == "train" else 2
    entries: List[Tuple[str, int]] = []
    for class_name, label in sorted(class_to_index.items(), key=lambda item: item[1]):
        split_file = os.path.join(
            split_root, f"{class_name}_test_split{split_index}.txt"
        )
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Missing HMDB split file: {split_file}")
        with open(split_file, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                parts = raw_line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    flag = int(parts[1])
                except ValueError:
                    continue
                if flag == target_flag:
                    relative = _normalize_path(os.path.join(class_name, parts[0]))
                    entries.append((relative, int(label)))
    return entries


class VPHMDBDataset(Dataset):
    def __init__(
        self,
        root: str,
        splits_root: str,
        privacy_label_pkl: str,
        action_classes_file: str,
        split: str,
        split_index: int,
        num_frames: int,
        image_size: int,
        training: bool,
        data_percentage: float = 1.0,
        seed: int = 42,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.num_frames = int(num_frames)
        self.training = bool(training)
        self.rng = random.Random(seed)
        classes = _hmdb_class_mapping(action_classes_file)
        entries = _hmdb_split_entries(
            splits_root,
            split_index,
            "train" if split == "train" else "val",
            classes,
        )
        privacy = _privacy_lookup(privacy_label_pkl)
        samples = []
        for relative, label in entries:
            class_name, basename = relative.split("/", 1)
            candidates = (
                os.path.join(root, relative),
                os.path.join(root, class_name, basename),
                os.path.join(root, class_name, class_name, basename),
            )
            absolute = next((item for item in candidates if os.path.isfile(item)), None)
            if absolute is None:
                continue
            stem = os.path.splitext(os.path.basename(relative))[0]
            mp4_relative = os.path.splitext(relative)[0] + ".mp4"
            vector = None
            for candidate in (
                _normalize_path(relative),
                mp4_relative,
                os.path.basename(relative),
                os.path.basename(mp4_relative),
                stem,
                _normalize_path(absolute),
            ):
                if candidate in privacy:
                    vector = privacy[candidate]
                    break
            if vector is not None:
                samples.append((absolute, int(label), vector.astype(np.float32)))
        if not samples:
            raise RuntimeError("VP-HMDB is empty; check dataset and label paths")
        if 0.0 < data_percentage < 1.0:
            self.rng.shuffle(samples)
            samples = samples[:max(1, round(len(samples) * data_percentage))]
        self.samples = samples
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _indices(self, frame_count: int) -> np.ndarray:
        if frame_count <= 0:
            raise RuntimeError("Video has no frames")
        if frame_count < self.num_frames:
            return np.linspace(0, frame_count - 1, self.num_frames).round().astype(np.int64)
        if not self.training:
            return np.linspace(0, frame_count - 1, self.num_frames).astype(np.int64)
        bins = np.linspace(0, frame_count, self.num_frames + 1).astype(int)
        return np.asarray([
            self.rng.randint(bins[index], max(bins[index] + 1, bins[index + 1]) - 1)
            for index in range(self.num_frames)
        ], dtype=np.int64)

    def __getitem__(self, index: int):
        video_path, action_label, privacy_label = self.samples[index]
        try:
            reader = decord.VideoReader(video_path, ctx=decord.cpu())
            frames = reader.get_batch(self._indices(len(reader)))
            frames = torch.stack([
                self.transform(frames[i].numpy()) for i in range(frames.shape[0])
            ])
        except Exception as exc:
            raise RuntimeError(f"Failed to read {video_path}: {exc}") from exc
        return (
            frames,
            torch.tensor(action_label, dtype=torch.long),
            torch.tensor(privacy_label, dtype=torch.float32),
            video_path,
        )


def make_loader(config, split: str, benchmark: bool = False) -> DataLoader:
    training = split == "train"
    dataset_class = VPUCFDataset if config.dataset.lower() == "vpucf" else VPHMDBDataset
    dataset = dataset_class(
        root=config.video_root,
        splits_root=config.split_root,
        privacy_label_pkl=config.privacy_label_pkl,
        action_classes_file=config.action_classes_file,
        split=split,
        split_index=config.split_index,
        num_frames=config.num_frames,
        image_size=config.image_size,
        training=training,
        data_percentage=config.data_percentage if training else 1.0,
        seed=config.seed,
    )
    if benchmark:
        batch_size = config.benchmark_batch_size if training else config.benchmark_val_batch_size
        workers = config.benchmark_num_workers
        prefetch = config.benchmark_prefetch_factor
    else:
        batch_size = config.adversarial_batch_size if training else config.adversarial_val_batch_size
        workers = config.num_workers
        prefetch = config.prefetch_factor if training else 1
    kwargs = dict(
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=True,
        drop_last=training,
    )
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch
        kwargs["persistent_workers"] = training
    return DataLoader(dataset, **kwargs)
