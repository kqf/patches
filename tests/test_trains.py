from pathlib import Path

from patches.train import main


def test_trains(train_valid_patches: tuple[Path, Path]):
    train, valid = train_valid_patches
    main(
        epochs=1,
        train_labels=str(train),
        valid_labels=str(valid),
    )
