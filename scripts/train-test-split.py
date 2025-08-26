from pathlib import Path

import click

from patches.dataset import (
    Patch,
    read_dataset,
    save_dataset,
    to_patches_dataset,
)


def train_test_split(patches: list[Patch]) -> tuple[list[Patch], list[Patch]]:
    train_labels = {
        "person",
        "truck",
    }
    train = [p for p in patches if p.label in train_labels]
    valid = [p for p in patches if p.label not in train_labels]
    return train, valid


@click.command()
@click.argument(
    "path",
    type=click.Path(
        exists=True,
        path_type=Path,
    ),
    default=Path(
        ".datasets/ground/VisDrone/VisDrone2019-DET-train/annotations.json",
    ),
)
def main(path: Path):
    patches = to_patches_dataset(read_dataset(path))
    save_dataset(str(path.with_stem("clean-patches")), patches)
    train, valid = train_test_split(patches)
    save_dataset(str(path.with_stem("clean-patches-train")), train)
    save_dataset(str(path.with_stem("clean-patches-valid")), valid)


if __name__ == "__main__":
    main()
