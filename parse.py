import json
from pathlib import Path

import click
import cv2

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
        img_file = img_dir / (ann_file.stem + ".jpg")
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


@click.command()
@click.argument(
    "path",
    type=click.Path(
        exists=True,
        file_okay=False,
        path_type=Path,
    ),
)
def main(path: Path):
    out_path = path / "annotations.json"
    result = parse_visdrone_annotations(path, out_path)
    click.echo(f"Annotations saved to {result}")


if __name__ == "__main__":
    main()
