import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler
from torchvision.datasets import MovingMNIST

from .transforms import apply_group


TRAIN_RANGE = (0, 8000)
VALIDATION_RANGE = (8000, 9000)
TEST_RANGE = (9000, 10000)
ALLOWED_START_COUNTS = {1: 14, 2: 8, 3: 2}


class VideoSplit(Dataset):
    def __init__(self, dataset, start, stop):
        self.dataset = dataset
        self.start = int(start)
        self.stop = int(stop)

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        item = self.dataset[self.start + index]
        video = item[0] if isinstance(item, tuple) else item
        video = torch.as_tensor(video)
        if video.ndim == 3:
            video = video.unsqueeze(1)
        if video.shape != (20, 1, 64, 64):
            raise ValueError(f"Unexpected Moving MNIST video shape: {tuple(video.shape)}")
        return video.to(torch.float32).div(255.0), self.start + index


def load_splits(root, download=False):
    dataset = MovingMNIST(root=str(root), split=None, download=download)
    return {
        "train": VideoSplit(dataset, *TRAIN_RANGE),
        "validation": VideoSplit(dataset, *VALIDATION_RANGE),
        "test": VideoSplit(dataset, *TEST_RANGE),
    }


def make_window(video, speed=1, start=1):
    speed = int(speed)
    start = int(start)
    indices = start - 1 + torch.arange(10) * speed
    input_indices = indices[:5]
    target_indices = indices[5:]
    if input_indices[-1] >= len(video):
        raise ValueError("The input window exceeds the video length")
    valid_target_indices = target_indices[target_indices < len(video)]
    inputs = video[input_indices]
    targets = torch.zeros((5, *video.shape[1:]), dtype=video.dtype)
    targets[: len(valid_target_indices)] = video[valid_target_indices]
    mask = torch.zeros(5, dtype=torch.bool)
    mask[: len(valid_target_indices)] = True
    return inputs, targets, mask


def frame_numbers(speed=1, start=1):
    speed = int(speed)
    start = int(start)
    values = start + torch.arange(10) * speed
    inputs = values[:5]
    targets = values[5:]
    return inputs.tolist(), targets[targets <= 20].tolist()


def sample_seed(seed, epoch, index):
    return int((seed + 1) * 1_000_003 + (epoch + 1) * 97_409 + (index + 1) * 65_537)


class EpochSampler(Sampler):
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 10_007)
        order = torch.randperm(len(self.dataset), generator=generator).tolist()
        return iter((self.epoch, index) for index in order)

    def __len__(self):
        return len(self.dataset)


class TrainingWindows(Dataset):
    def __init__(self, videos, model_name, seed):
        self.videos = videos
        self.model_name = model_name.upper()
        self.seed = int(seed)
        if self.model_name not in {"A", "B", "C", "D", "DB"}:
            raise ValueError(self.model_name)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, key):
        epoch, index = key
        video, video_index = self.videos[index]
        generator = torch.Generator().manual_seed(sample_seed(self.seed, epoch, index))
        speed_augmented = self.model_name in {"C", "D", "DB"}
        symmetry_augmented = self.model_name in {"B", "DB"}
        speed = int(torch.randint(1, 4, (1,), generator=generator).item()) if speed_augmented else 1
        start_count = ALLOWED_START_COUNTS[speed] if speed_augmented else 11
        start = int(torch.randint(1, start_count + 1, (1,), generator=generator).item())
        inputs, targets, mask = make_window(video, speed, start)
        group = int(torch.randint(0, 8, (1,), generator=generator).item()) if symmetry_augmented else 0
        if symmetry_augmented:
            inputs = apply_group(inputs, group)
            targets = apply_group(targets, group)
        return {
            "inputs": inputs,
            "targets": targets,
            "mask": mask,
            "video_index": video_index,
            "speed": speed,
            "start": start,
            "group": group,
        }


class FixedWindows(Dataset):
    def __init__(self, videos, speed=1, start=1, group=0):
        self.videos = videos
        self.speed = int(speed)
        self.start = int(start)
        self.group = int(group)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video, video_index = self.videos[index]
        inputs, targets, mask = make_window(video, self.speed, self.start)
        if self.group:
            inputs = apply_group(inputs, self.group)
            targets = apply_group(targets, self.group)
        return {
            "inputs": inputs,
            "targets": targets,
            "mask": mask,
            "video_index": video_index,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    splits = load_splits(args.root, download=args.download)
    print({name: len(split) for name, split in splits.items()}, flush=True)


if __name__ == "__main__":
    main()

