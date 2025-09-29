import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from patches.parse import CATEGORY_MAP, parse_visdrone_annotations


@pytest.fixture
def raw_dataset(
    use_real: bool,
    tmp_path: Path,
    resolution: tuple[int, int] = (640, 480),
    num_samples=3,
) -> Path:
    if use_real:
        return Path(".datasets/ground/VisDrone/VisDrone2019-DET-train/")

    visdrone_path = tmp_path / "fake-visdrone"

    img_dir = visdrone_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = visdrone_path / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        image = np.random.randint(0, 256, (*resolution, 3), dtype=np.uint8)
        img_file = img_dir / f"{i:03d}.jpg"
        cv2.imwrite(str(img_file), image)

        # Generate fake annotation lines
        lines = []
        for _ in range(random.randint(1, 3)):  # 1-3 annotations per image
            x = random.randint(0, resolution[0] - 50)
            y = random.randint(0, resolution[1] - 50)
            w = random.randint(20, 50)
            h = random.randint(20, 50)
            score = random.randint(0, 1)
            cat = random.choice(list(CATEGORY_MAP.keys()))
            trunc = random.randint(0, 1)
            occ = random.randint(0, 1)
            line = f"{x},{y},{w},{h},{score},{cat},{trunc},{occ}"
            lines.append(line)

        ann_file = ann_dir / f"{i:03d}.txt"
        ann_file.write_text("\n".join(lines))

    return visdrone_path


@pytest.mark.parametrize("use_real", [False])
def test_parses(raw_dataset: Path):
    out_path = raw_dataset / "annotations.json"
    result = parse_visdrone_annotations(raw_dataset, out_path)
    print(f"Annotations saved to {result}")
