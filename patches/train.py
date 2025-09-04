from dataclasses import dataclass
from functools import partial

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from torchvision.models import ResNet50_Weights, resnet50

from patches.pipeline import LocalizationPipeline


@dataclass
class Config:
    tracking_uri: str = "dummy-server"
    datapath: str = ".datasets/ground/"


def build_model() -> torch.nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    return model


def main(
    run_name: str = "first-run",
    # NB: they are swapped intentionally
    train_labels: str = "VisDrone/VisDrone2019-DET-train/clean-patches-valid.json",  # noqa
    valid_labels: str = "VisDrone/VisDrone2019-DET-train/clean-patches-train.json",  # noqa
    resolution: tuple[int, int] = (128, 128),
    epochs: int = 10,
    batch_size=16,
) -> None:
    config = Config()
    pl.trainer.seed_everything(137)  # type: ignore
    pipeline = LocalizationPipeline(
        train_labels=config.datapath + train_labels,
        valid_labels=config.datapath + valid_labels,
        model=build_model(),
        resolution=resolution,
        build_optimizer=partial(
            torch.optim.Adam,
            # lr=0.01,
            # weight_decay=0.0001,
            # momentum=0.9,
        ),
        build_scheduler=partial(
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            T_0=10,
            T_mult=2,
        ),
        loss=torch.nn.SmoothL1Loss(),
        batch_size=batch_size,
    )

    print("Cuda is available: ", torch.cuda.is_available())
    trainer = pl.Trainer(
        # gpus=4,
        # amp_level=O1,
        # devices=8,
        max_epochs=epochs,
        strategy="ddp_find_unused_parameters_true",
        num_sanity_val_steps=0,
        benchmark=True,
        precision=16,
        sync_batchnorm=torch.cuda.is_available(),
        logger=MLFlowLogger(
            save_dir="mlruns",
            # tracking_uri=config.tracking_uri,
            experiment_name="patches",
            run_name=run_name,
        ),
        callbacks=[
            ModelCheckpoint(
                monitor="valid_loss_epoch",
                verbose=True,
                mode="min",
                save_top_k=-1,
                save_weights_only=True,
            ),
            TQDMProgressBar(
                refresh_rate=10,
            ),
        ],
    )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()
