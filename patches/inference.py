from os import PathLike
from pathlib import Path
from typing import Callable

import albumentations as alb
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from patches.train import build_model

XYWH = tuple[int, int, int, int]
XYXY = tuple[float, float, float, float]


def test(resolution: tuple[int, int]) -> alb.Compose:
    return alb.Compose(
        [
            alb.PadIfNeeded(
                min_width=resolution[0],
                min_height=resolution[1],
                border_mode=0,
            ),
            alb.Resize(width=resolution[0], height=resolution[1]),
            ToTensorV2(),
        ]
    )


def load_state_dict(checkpoint: str | Path | PathLike) -> dict:
    state_dict = torch.load(
        checkpoint,
        map_location=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )
    return {
        k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()
    }


def crop(frame: np.ndarray, bbox: XYWH) -> np.ndarray:
    x, y, w, h = bbox
    return frame[y : y + h, x : x + w]


def plot(image: np.ndarray, bbox: XYXY) -> np.ndarray:
    h, w, _ = image.shape
    x_min, y_min, x_max, y_max = bbox
    return cv2.rectangle(
        image.copy(),
        (int(x_min * w), int(y_min * h)),
        (int(x_max * w), int(y_max * h)),
        color=(0, 255, 0),
        thickness=2,
    )


def build_inference(
    checkpoint: str | Path | PathLike, size: int
) -> Callable[[np.ndarray], XYXY]:
    transform = test(resolution=(size, size))
    model = build_model()
    if checkpoint:
        model.load_state_dict(load_state_dict(checkpoint))
    model.eval()

    def run_inference(batch: torch.Tensor):
        batch = batch.unsqueeze(0).to(
            dtype=torch.float32, device=next(model.parameters()).device
        )
        with torch.no_grad():
            return model(batch)

    def infer(image: np.ndarray) -> XYXY:
        batch = transform(image=image)["image"]
        return run_inference(batch)[0].cpu().numpy().tolist()

    return infer


def to_global(patch: XYXY, frame: XYWH) -> XYWH:
    xmin, ymin, xmax, ymax = patch
    x, y, w, h = frame
    gx = int(x + xmin * w)
    gy = int(y + ymin * h)
    gw = int((xmax - xmin) * w)
    gh = int((ymax - ymin) * h)
    return gx, gy, gw, gh


def infer(
    frame: np.ndarray,
    bbox: XYWH,
    predictor: Callable,
    size=120,
) -> tuple:
    roi = crop(frame, bbox)
    roi_resized = cv2.resize(roi, (size, size))
    local_bbox = predictor(roi_resized)
    global_bbox = to_global(local_bbox, bbox)
    return frame, global_bbox, roi_resized, local_bbox


def plot_all(
    frame: np.ndarray,
    bbox: XYWH,
    roi_resized: np.ndarray,
    local_bbox: XYXY,
) -> np.ndarray:
    x, y, w, h = bbox
    frame = cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), (255, 0, 0), 2)
    vis = plot(roi_resized, local_bbox)
    h_patch, w_patch = vis.shape[:2]
    h_frame, w_frame = frame.shape[:2]
    x_offset = w_frame - w_patch
    y_offset = h_frame - h_patch
    frame[y_offset : y_offset + h_patch, x_offset : x_offset + w_patch] = vis
    return frame


def main(
    iimage: str = "image.jpeg",
    oimage: str = "predicted.jpeg",
    checkpoint: str = "222.ckpt",
):
    frame = cv2.imread(iimage)
    bbox = (20, 20, 50, 50)
    predictor = build_inference(checkpoint, 120)
    frame, newbox, roi, locbox = infer(frame, bbox, predictor)
    frame = plot_all(frame, newbox, roi, locbox)
    cv2.imwrite(oimage, frame)
    cv2.imshow("Prediction", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
