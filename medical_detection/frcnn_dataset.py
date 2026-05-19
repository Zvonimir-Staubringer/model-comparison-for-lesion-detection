from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T_v2

from .dataset import ImageRecord
from .detection import build_frcnn_target


@dataclass(frozen=True)
class FRCNNDatasetConfig:
    include_empty_images: bool = True
    train: bool = False


def build_image_transform(train: bool) -> T_v2.Compose:
    transforms: list[torch.nn.Module] = [T_v2.ToImage(), T_v2.ToDtype(torch.float32, scale=True)]
    if train:
        transforms.append(T_v2.ColorJitter(brightness=0.1, contrast=0.1))
    return T_v2.Compose(transforms)


class FRCNNDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(self, records: list[ImageRecord], config: FRCNNDatasetConfig):
        if config.include_empty_images:
            self.records = records
        else:
            self.records = [record for record in records if record.has_annotations]
        self.transform = build_image_transform(train=config.train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        image_width, image_height = image.size
        target = build_frcnn_target(record.annotations, image_width, image_height, image_id=index)
        image_tensor = self.transform(image)
        return image_tensor, target


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets