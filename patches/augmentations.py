from typing import Tuple

import albumentations as alb
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

RelativeXYXY = Tuple[float, float, float, float]


def to_numpy(
    image: torch.Tensor,
    bbox: torch.Tensor,
) -> Tuple[np.ndarray, RelativeXYXY]:
    # Convert image tensor to numpy HWC
    image_np = image.detach().cpu().numpy()  # C,H,W
    image_np = np.transpose(image_np, (1, 2, 0))  # H,W,C
    bbox_rel: RelativeXYXY = tuple(bbox.detach().cpu().numpy().tolist())  # type: ignore
    return image_np, bbox_rel


def transform(
    image: np.ndarray,
    bbox: RelativeXYXY,
    pipeline: alb.Compose,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = image.shape[:2]

    # albumentations expects absolute coords for bboxes
    abs_bbox = [
        int(bbox[0] * w),
        int(bbox[1] * h),
        int(bbox[2] * w),
        int(bbox[3] * h),
    ]

    transformed = pipeline(
        image=image,
        bboxes=[abs_bbox],
        cropping_bbox=abs_bbox,  # for RandomCropNearBBox
        labels=[0],  # dummy label
    )

    transformed_image = transformed["image"]
    transformed_bbox = transformed["bboxes"][0]

    # C,H,W
    h, w = transformed_image.shape[1:]
    rel_bbox = torch.tensor(
        [
            transformed_bbox[0] / w,
            transformed_bbox[1] / h,
            transformed_bbox[2] / w,
            transformed_bbox[3] / h,
        ],
        dtype=torch.float32,
    )

    return transformed_image, rel_bbox


def train(resolution: tuple[int, int]) -> alb.Compose:
    return alb.Compose(
        [
            alb.PadIfNeeded(
                min_width=2 * resolution[0],
                min_height=2 * resolution[1],
                border_mode=cv2.BORDER_CONSTANT,
            ),
            alb.RandomSizedBBoxSafeCrop(
                width=resolution[0],
                height=resolution[1],
                erosion_rate=0.0,
                p=1.0,
            ),
            alb.HorizontalFlip(p=0.5),
            alb.RandomBrightnessContrast(p=0.3),
            alb.MotionBlur(p=0.2),
            ToTensorV2(),
        ],
        bbox_params=alb.BboxParams(
            format="pascal_voc", label_fields=["labels"]
        ),
    )


def valid(resolution: tuple[int, int]) -> alb.Compose:
    return alb.Compose(
        [
            alb.PadIfNeeded(
                min_width=resolution[0],
                min_height=resolution[1],
                border_mode=0,
            ),
            alb.Resize(
                width=resolution[0],
                height=resolution[1],
            ),
            ToTensorV2(),
        ],
        bbox_params=alb.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
        ),
    )
