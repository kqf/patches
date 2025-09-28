import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type, TypeVar

import albumentations as alb
import cv2
import torch
import torch.utils
import torch.utils.data
from dacite import Config, from_dict
from dataclasses_json import dataclass_json

from patches.augmentations import transform

# Your schema
RelativeXYXY = tuple[float, float, float, float]


def save_dataset(out_path: str | Path, samples: list) -> Path:
    # Save to JSON using dataclasses_json
    with open(out_path, "w") as f:
        json.dump([s.to_dict() for s in samples], f, indent=2)
    return Path(out_path)


@dataclass_json
@dataclass
class Annotation:
    bbox: RelativeXYXY
    landmarks: list = field(default_factory=list)
    label: str = "person"
    score: float = float("nan")
    truncation: int = 0
    occlusion: int = 0


@dataclass_json
@dataclass
class Sample:
    file_name: str
    annotations: list[Annotation]


@dataclass_json
@dataclass
class Patch:
    file_name: str
    bbox: RelativeXYXY
    landmarks: list = field(default_factory=list)
    label: str = "person"
    score: float = float("nan")
    truncation: int = 0
    occlusion: int = 0
    n: int = 3

    def read(self, border_type: int = cv2.BORDER_REPLICATE):
        image = cv2.imread(self.file_name)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {self.file_name}")

        h, w = image.shape[:2]
        x1, y1, x2, y2 = self.bbox

        # scale bbox to absolute pixel coords
        abs_x1, abs_y1, abs_x2, abs_y2 = (
            int(x1 * w),
            int(y1 * h),
            int(x2 * w),
            int(y2 * h),
        )

        bw, bh = abs_x2 - abs_x1, abs_y2 - abs_y1
        max_size = max(bw, bh)

        # crop region centered at bbox
        cx, cy = (abs_x1 + abs_x2) // 2, (abs_y1 + abs_y2) // 2
        half = (max_size * self.n) // 2

        crop_x1 = cx - half
        crop_y1 = cy - half
        crop_x2 = cx + half
        crop_y2 = cy + half

        # calculate required padding
        pad_left = max(0, -crop_x1)
        pad_top = max(0, -crop_y1)
        pad_right = max(0, crop_x2 - w)
        pad_bottom = max(0, crop_y2 - h)

        if any([pad_left, pad_top, pad_right, pad_bottom]):
            image = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right, border_type
            )
            h, w = image.shape[:2]
            # shift crop box because we padded
            crop_x1 += pad_left
            crop_y1 += pad_top
            crop_x2 += pad_left
            crop_y2 += pad_top

        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

        # recalc bbox relative to cropped image
        new_x1 = (abs_x1 + pad_left - crop_x1) / (crop_x2 - crop_x1)
        new_y1 = (abs_y1 + pad_top - crop_y1) / (crop_y2 - crop_y1)
        new_x2 = (abs_x2 + pad_left - crop_x1) / (crop_x2 - crop_x1)
        new_y2 = (abs_y2 + pad_top - crop_y1) / (crop_y2 - crop_y1)

        new_bbox = (new_x1, new_y1, new_x2, new_y2)

        new_ann = Annotation(
            bbox=new_bbox,
            landmarks=self.landmarks,
            label=self.label,
            score=self.score,
            truncation=self.truncation,
            occlusion=self.occlusion,
        )
        return cropped, new_ann


T = TypeVar(
    "T",
    Sample,
    Patch,
)


def to_sample(entry: dict[str, Any], data_class: Type[T]) -> T:
    try:
        return from_dict(
            data_class=data_class,
            data=entry,
            config=Config(cast=[tuple]),
        )
    except Exception as e:
        print(f"Failed to parse entry: {entry}")
        raise e


def read_dataset(path: Path | str) -> list[Sample]:
    path = Path(path)
    with open(path) as f:
        df = json.load(f)
    samples = [to_sample(x, data_class=Sample) for x in df if x]
    for sample in samples:
        sample.annotations = [
            a for a in sample.annotations if a.bbox is not None
        ]
    return [s for s in samples if len(s.annotations) > 0]


def read_patches_dataset(path: Path | str) -> list[Patch]:
    path = Path(path)
    with open(path) as f:
        df = json.load(f)
    return [to_sample(x, data_class=Patch) for x in df if x]


def is_on_edge(annotation: Annotation, eps: float = 1e-6) -> bool:
    x1, y1, x2, y2 = annotation.bbox
    return (
        abs(x1) < eps
        or abs(y1) < eps
        or abs(1 - x2) < eps
        or abs(1 - y2) < eps
    )


def iou(b1: RelativeXYXY, b2: RelativeXYXY) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def has_overlaps(
    annotation: Annotation,
    annotations: list[Annotation],
    iou_thr: float = 0.0,
) -> bool:
    for other in annotations:
        if other is annotation:
            continue
        if iou(annotation.bbox, other.bbox) > iou_thr:
            return True
    return False


def patch_box(bbox: RelativeXYXY, n: int) -> RelativeXYXY:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1
    max_size = max(bw, bh)
    half = (max_size * n) / 2
    return (cx - half, cy - half, cx + half, cy + half)


def has_other_objects_on_patch(
    annotation: Annotation,
    annotations: list[Annotation],
    n: int,
    iou_thr: float = 0.0,
) -> bool:
    patch_region = patch_box(annotation.bbox, n)
    for other in annotations:
        if other is annotation:
            continue
        if iou(patch_region, other.bbox) > iou_thr:
            return True
    return False


def to_patches_dataset(samples: list[Sample], n: int = 3) -> list[Patch]:
    patches = []
    for sample in samples:
        for ann in sample.annotations:
            # Check if bouding box touches the image edges -> then ignore
            # because reflect will be broken
            if is_on_edge(ann):
                continue

            # Check if our annotation overlaps with any other annotation
            if has_overlaps(ann, sample.annotations):
                continue

            # Even stricter check if any other object will
            # appear in the final crop
            if has_other_objects_on_patch(ann, sample.annotations, n):
                continue

            patches.append(
                Patch(
                    file_name=sample.file_name,
                    bbox=ann.bbox,
                    landmarks=ann.landmarks,
                    label=ann.label,
                    score=ann.score,
                    truncation=ann.truncation,
                    occlusion=ann.occlusion,
                    n=n,
                )
            )
    return patches


class PatchesDataset(torch.utils.data.Dataset):
    def __init__(self, label_path: str | Path, pipeline: alb.Compose):
        self.pipeline = pipeline
        self.patches = read_patches_dataset(label_path)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[idx]
        oimage, annotation = patch.read()
        try:
            image, bbox = transform(oimage, annotation.bbox, self.pipeline)
        except TypeError:
            print("Failed at", idx)
            image, bbox = transform(oimage, (0, 0, 1.0, 1.0), self.pipeline)
        return image, bbox
