from collections import Counter
from functools import partial

import cv2

from patches.augmentations import to_numpy, train, transform, valid
from patches.dataset import read_patches_dataset
from patches.plot import plot


def main():
    # patches = to_patches_dataset(
    #     read_dataset(
    #         ".datasets/ground/VisDrone/VisDrone2019-DET-train/annotations.json"
    #     )
    # )
    patches = read_patches_dataset(
        ".datasets/ground/VisDrone/VisDrone2019-DET-train/clean-patches-train.json"  # noqa
    )
    augment = partial(
        transform,
        pipeline=train(resolution=(128, 128)),
    )
    augment = partial(
        transform,
        pipeline=valid(resolution=(128, 128)),
    )

    print(len(patches))
    labels = Counter()
    for patch in patches:
        image, annotation = patch.read()
        labels[annotation.label] += 1
        timage, annotation.bbox = to_numpy(*augment(image, annotation.bbox))
        print(annotation.bbox)
        print(image.shape, "->", timage.shape)
        cv2.imshow("frame", plot(timage, [annotation]))
        cv2.waitKey()
    print(labels)


if __name__ == "__main__":
    main()
