import cv2
import pytest

from patches.dataset import read_dataset


@pytest.fixture
def dataset(annotations, use_real=False):
    if not use_real:
        return read_dataset(annotations)
    return read_dataset(
        ".datasets/ground/VisDrone/VisDrone2019-DET-train/annotations.json"
    )


def test_dataset(dataset):
    for sample in dataset:
        image = cv2.imread(sample.file_name)
        if image is None:
            continue
        # cv2.imshow("frame", plot(image, sample.annotations))
        # cv2.waitKey()
    print(len(dataset))
