from pathlib import Path

import pytest

from patches.train import main


def patches_dataset(tmp_path: Path) -> Path:
    return tmp_path / "patches.json"


@pytest.mark.skip
def test_trains(patches_dataset: Path):
    main(
        epochs=1,
        train_labels=str(patches_dataset),
        valid_labels=str(patches_dataset),
    )
