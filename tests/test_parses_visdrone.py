import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from patches.dataset import Annotation, Sample


def parse(line: str):
    line = line.strip()
    if line.strip().endswith(","):
        line = line[:-1]
    return map(int, line.strip().split(","))


CATEGORY_MAP = {
    1: "pedestrian",
    2: "person",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


def parse_visdrone_annotations(dataset_dir: Path, out_path: Path):
    dataset_dir = Path(dataset_dir)
    ann_dir = dataset_dir / "annotations"
    img_dir = dataset_dir / "images"

    samples: list[Sample] = []

    for ann_file in sorted(ann_dir.glob("*.txt")):
        img_file = img_dir / f"{ann_file.stem}.jpg"
        if not img_file.exists():
            continue

        # Get image size for normalization using OpenCV
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w = img.shape[:2]

        anns: list[Annotation] = []
        with open(ann_file) as f:
            for line in f:
                x, y, bw, bh, score, cat, trunc, occ = parse(line)

                if cat not in CATEGORY_MAP:
                    continue

                # Convert to relative xyxy
                x1, y1 = x / w, y / h
                x2, y2 = (x + bw) / w, (y + bh) / h

                anns.append(
                    Annotation(
                        bbox=(x1, y1, x2, y2),
                        label=CATEGORY_MAP[cat],
                        score=float(score) if score >= 0 else float("nan"),
                        truncation=trunc,
                        occlusion=occ,
                    )
                )

        samples.append(Sample(file_name=str(img_file), annotations=anns))

    # Save to JSON using dataclasses_json
    with open(out_path, "w") as f:
        json.dump([s.to_dict() for s in samples], f, indent=2)  # type: ignore

    return out_path


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
