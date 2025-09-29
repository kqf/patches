from functools import partial

import cv2
import pytest

from patches.augmentations import to_numpy, transform, valid
from patches.dataset import Patch, read_patches_dataset
from patches.plot import plot
from patches.train import build_model


@pytest.fixture
def patches(use_real: bool) -> list[Patch]:
    if use_real:
        return read_patches_dataset(
            ".datasets/ground/VisDrone/VisDrone2019-DET-train/clean-patches-train.json"  # noqa
        )
    return []


@pytest.mark.parametrize("use_real", [False])
def test_augments(patches: list[Patch]):
    model = build_model()
    model.eval()
    augment = partial(
        transform,
        pipeline=valid(resolution=(128, 128)),
    )

    print(len(patches))
    # sourcery skip: no-loop-in-tests
    for patch in patches:
        image, annotation = patch.read()
        batch, bbox = augment(image, annotation.bbox)
        pbox = model(batch)
        print(pbox, bbox)
        timage, annotation.bbox = to_numpy(*augment(image, annotation.bbox))
        print(annotation.bbox)
        print(image.shape, "->", timage.shape)
        cv2.imshow("frame", plot(timage, [annotation]))
        cv2.waitKey()
