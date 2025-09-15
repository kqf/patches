from pathlib import Path

import cv2
import pytest

from patches.dataset import Sample, read_dataset
from patches.plot import plot


@pytest.fixture
def dataset(annotations: Path, use_real: bool) -> list[Sample]:
    if not use_real:
        return read_dataset(annotations)

    return read_dataset(
        ".datasets/ground/VisDrone/VisDrone2019-DET-train/annotations.json"
    )


@pytest.mark.parametrize("use_real", [True])
def test_dataset(dataset: list[Sample], use_real: bool):
    # sourcery skip: no-loop-in-tests
    for sample in dataset:
        image = cv2.imread(sample.file_name)
        # sourcery skip: no-conditionals-in-tests
        if image is None:
            continue
        if use_real:
            cv2.imshow("frame", plot(image, sample.annotations))
            cv2.waitKey()
    print(len(dataset))
