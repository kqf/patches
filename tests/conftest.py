import json
import pathlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from patches.dataset import (
    read_dataset,
    save_dataset,
    to_patches_dataset,
)


def pytest_addoption(parser):
    parser.addoption(
        "--use-real",
        action="store_true",
        default=None,
        help="Force use_real=True in parametrized tests",
    )


def pytest_generate_tests(metafunc):
    if "use_real" in metafunc.fixturenames:
        cli_value = metafunc.config.getoption("--use-real")
        if cli_value is not None:
            metafunc.parametrize("use_real", [cli_value])


@pytest.fixture
def annotations(
    tmp_path: pathlib.Path,
    resolution: tuple[int, int] = (640, 480),
) -> pathlib.Path:
    image = np.random.randint(0, 256, (*resolution, 3), dtype=np.uint8)
    image_path = str(tmp_path / "image.png")
    cv2.imwrite(image_path, image)
    example = [
        {
            "file_name": image_path,
            "annotations": [
                {
                    "bbox": [
                        0.7375,
                        0.8722,
                        0.8145,
                        0.9333,
                    ],
                    "landmarks": [],
                    "label": "car",
                    "score": 1.0,
                    "truncation": 0,
                    "occlusion": 1,
                },
                {
                    "bbox": [
                        0.665625,
                        0.787037,
                        0.729166,
                        0.872222,
                    ],
                    "landmarks": [],
                    "label": "person",
                    "score": 1.0,
                    "truncation": 0,
                    "occlusion": 0,
                },
                {
                    "bbox": [
                        0.61875,
                        0.73888,
                        0.68541,
                        0.83333,
                    ],
                    "landmarks": [],
                    "label": "car",
                    "score": 1.0,
                    "truncation": 0,
                    "occlusion": 0,
                },
            ],
        }
    ]
    ofile = tmp_path / "annotations.json"
    with open(ofile, "w") as f:
        json.dump(example, f, indent=4)
    return ofile


def train_valid_patches(annotations: Path) -> tuple[Path, Path]:
    patches = to_patches_dataset(read_dataset(annotations))
    train_labels = {
        "person",
        "truck",
    }
    train = [p for p in patches if p.label in train_labels]
    valid = [p for p in patches if p.label not in train_labels]
    return (
        save_dataset(str(annotations.with_stem("clean-patches-train")), train),
        save_dataset(str(annotations.with_stem("clean-patches-valid")), valid),
    )
