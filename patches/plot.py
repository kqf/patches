import cv2
import numpy as np

from patches.dataset import Annotation


def plot(
    image: np.ndarray,
    annotations: list[Annotation],
    true=True,
) -> np.ndarray:
    plotted = image.copy()
    color = (0, 255, 0) if true else (0, 0, 255)
    h, w, _ = image.shape
    for annotation in annotations:
        for x, y in annotation.landmarks:
            plotted = cv2.circle(
                plotted,
                (int(x * h), int(y * w)),
                radius=1,
                color=color,
                thickness=1,
            )

        x_min, y_min, x_max, y_max = (tx for tx in annotation.bbox)

        plotted = cv2.rectangle(
            plotted,
            (int(x_min * w), int(y_min * h)),
            (int(x_max * w), int(y_max * h)),
            color=color,
            thickness=2,
        )

        # Class label text underneath the box
        label_position = (int(x_min * w), int(y_max * h) + 15)
        plotted = cv2.putText(
            plotted,
            annotation.label,
            label_position,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return plotted
