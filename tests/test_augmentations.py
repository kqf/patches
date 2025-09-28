from collections import Counter
from functools import partial

import cv2
import pytest

from patches.augmentations import to_numpy, train, transform, valid
from patches.dataset import Patch, read_patches_dataset
from patches.plot import plot


@pytest.fixture
def patches(use_real: bool) -> list[Patch]:
    if use_real:
        return read_patches_dataset(
            ".datasets/ground/VisDrone/VisDrone2019-DET-train/clean-patches-train.json"  # noqa
        )
    return []


@pytest.mark.parametrize("use_real", [False])
def test_augments(patches: list[Patch]):
    augment = partial(
        transform,
        pipeline=train(resolution=(128, 128)),
    )
    augment = partial(
        transform,
        pipeline=valid(resolution=(128, 128)),
    )

    print(len(patches))
    labels: Counter = Counter()
    # sourcery skip: no-loop-in-tests
    for patch in patches:
        image, annotation = patch.read()
        labels[annotation.label] += 1
        timage, annotation.bbox = to_numpy(*augment(image, annotation.bbox))
        print(annotation.bbox)
        print(image.shape, "->", timage.shape)
        cv2.imshow("frame", plot(timage, [annotation]))
        cv2.waitKey()
    print(labels)
