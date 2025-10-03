from os import PathLike
from pathlib import Path
from typing import Callable, Optional

import albumentations as alb
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from patches.train import build_model

XYWH = tuple[int, int, int, int]  # absolute coords
XYXY = tuple[float, float, float, float]  # relative coords


def test(resolution: tuple[int, int]) -> alb.Compose:
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
    )


def load_state_dict(checkpiont: str | Path | PathLike) -> dict:
    state_dict = torch.load(
        checkpiont,
        map_location=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )
    return {
        k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()
    }


def plot(image, bbox):
    h, w, _ = image.shape
    x_min, y_min, x_max, y_max = bbox
    return cv2.rectangle(
        image,
        (int(x_min * w), int(y_min * h)),
        (int(x_max * w), int(y_max * h)),
        color=(0, 255, 0),
        thickness=2,
    )


def build_inference(
    checkpoint: str | Path | PathLike,
    size: int,
) -> Callable[[np.ndarray], tuple[float, float, float, float]]:
    transform = test(resolution=(size, size))
    model = build_model()
    model.load_state_dict(load_state_dict(checkpoint))

    def run_inference(model: torch.nn.Module, batch: torch.Tensor):
        batch = batch.unsqueeze(0)

        batch = batch.to(
            dtype=torch.float32, device=next(model.parameters()).device
        )

        model.eval()
        with torch.no_grad():
            return model(batch)

    def infer(image: np.ndarray) -> tuple[float, float, float, float]:
        batch = transform(image=image)["image"]
        return run_inference(model, batch)[0].cpu().numpy()

    return infer


def to_global(patch: XYXY, frame: XYWH) -> XYWH:
    xmin, ymin, xmax, ymax = patch
    x, y, w, h = frame
    gx = int(x + xmin * w)
    gy = int(y + ymin * h)
    gw = int((xmax - xmin) * w)
    gh = int((ymax - ymin) * h)
    return gx, gy, gw, gh


def infer(frame: np.ndarray, bbox: XYWH, infer: Callable, size=120) -> XYWH:
    # Crop first
    roi = crop(frame, bbox)

    # Crop and resize the expanded region
    roi_resized = cv2.resize(roi, (size, size))

    # Run inference
    local_bbox = infer(roi_resized)

    global_bbox = to_global(.local_bbox, bbox_roi)
    return frame, global_bbox, roi_resized, local_bbox


def plot_all(frame: np.ndarray, bbox: XYWH, roi_resized: np.ndarray, local_bbox: XYWH) -> np.ndarray:
    vis = plot(roi_resized, local_bbox)

    # Add plotted patch in the bottom right corner of the image
    h_patch, w_patch = vis.shape[:2]
    h_frame, w_frame = frame.shape[:2]

    # Calculate position for the patch (bottom right)
    x_offset = w_frame - w_patch
    y_offset = h_frame - h_patch

    # Overlay the patch
    frame[y_offset : y_offset + h_patch, x_offset : x_offset + w_patch] = (
        vis
    )
    return frame


def main()
    frame = cv2.imread("image.jpeg")
    bbox = (20, 20, 50, 50)
    predict = build_inference('222.ckpt', 120)
    frame, newbox, roi, locbox = infer(frame, bbox, predict)
    frame = plot_all(frame, newbox, roi, locbox)
    # Plot original bbox with differnt color
    cv2.imwrite("predicted.jpeg", frame)
    cv2.imshow()
    cv2.waitKey()


if __name__ == "__main__":
    main()
