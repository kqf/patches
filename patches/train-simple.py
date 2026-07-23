from dataclasses import dataclass

import mlflow
import torch
from skorch import NeuralNet
from skorch.callbacks import Checkpoint, LRScheduler, MlflowLogger, ProgressBar
from skorch.helper import predefined_split
from torchvision.models import ResNet50_Weights, resnet50

import patches.augmentations as augs
from patches.dataset import PatchesDataset


@dataclass
class Config:
    tracking_uri: str = "dummy-server"
    datapath: str = ".datasets/ground/"


class LocalizationModel(torch.nn.Module):
    """ResNet50 head that casts the (possibly uint8) input to float."""

    def __init__(self) -> None:
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 4)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.float())


def build_model() -> torch.nn.Module:
    return LocalizationModel()


def main(
    run_name: str = "first-run",
    # NB: they are swapped intentionally
    train_labels: str = "VisDrone/VisDrone2019-DET-train/clean-patches-valid.json",  # noqa
    valid_labels: str = "VisDrone/VisDrone2019-DET-train/clean-patches-train.json",  # noqa
    resolution: tuple[int, int] = (128, 128),
    epochs: int = 10,
    batch_size=16,
    config: Config | None = None,
) -> None:
    config = config or Config()
    torch.manual_seed(137)

    train_dataset = PatchesDataset(
        label_path=config.datapath + train_labels,
        pipeline=augs.train(resolution),
    )
    valid_dataset = PatchesDataset(
        label_path=config.datapath + valid_labels,
        pipeline=augs.valid(resolution),
    )

    print("Cuda is available: ", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("patches")

    with mlflow.start_run(run_name=run_name) as run:
        net = NeuralNet(
            module=build_model(),
            criterion=torch.nn.SmoothL1Loss,
            optimizer=torch.optim.Adam,
            max_epochs=epochs,
            batch_size=batch_size,
            train_split=predefined_split(valid_dataset),
            iterator_train__shuffle=True,
            iterator_train__num_workers=12,
            iterator_train__pin_memory=True,
            iterator_valid__shuffle=False,
            iterator_valid__num_workers=12,
            iterator_valid__pin_memory=True,
            device=device,
            callbacks=[
                (
                    "lr_scheduler",
                    LRScheduler(
                        policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,  # noqa
                        T_0=10,
                        T_mult=2,
                    ),
                ),
                (
                    "checkpoint",
                    Checkpoint(
                        monitor="valid_loss_best",
                        f_params="best_params.pt",
                    ),
                ),
                ("mlflow", MlflowLogger(run=run)),
                ("progress", ProgressBar()),
            ],
        )

        net.fit(train_dataset, y=None)


if __name__ == "__main__":
    main()
