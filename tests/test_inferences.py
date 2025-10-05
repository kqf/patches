from pathlib import Path

import cv2
import numpy as np
import pytest

from patches.inference import main


@pytest.fixture
def frame(use_real: bool, tmp_path: Path) -> str:
    # sourcery skip: no-conditionals-in-tests
    if use_real:
        return "image.jpeg"

    # Synthetic 640x480 image
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    output = str(tmp_path / "testimage.jpg")
    cv2.imwrite(output, frame)
    return output


@pytest.mark.parametrize("use_real", [False])
def test_inferences(frame):
    main(iimage=frame, checkpoint="")
