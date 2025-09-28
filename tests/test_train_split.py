from pathlib import Path

import pytest

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


@pytest.fixture
def test_splits(annotations: Path) -> None:
    patches = to_patches_dataset(read_dataset(annotations))
    save_dataset(str(annotations.with_stem("clean-patches")), patches)
    train, valid = train_test_split(patches)
    save_dataset(str(annotations.with_stem("clean-patches-train")), train)
    save_dataset(str(annotations.with_stem("clean-patches-valid")), valid)
