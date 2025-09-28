import cv2

from patches.dataset import read_dataset
from patches.plot import plot


def main():
    dataset = read_dataset(
        ".datasets/ground/VisDrone/VisDrone2019-DET-train/annotations.json"
    )
    for sample in dataset:
        image = cv2.imread(sample.file_name)
        if image is None:
            continue
        cv2.imshow("frame", plot(image, sample.annotations))
        cv2.waitKey()

    print(len(dataset))


if __name__ == "__main__":
    main()
