import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import patches.augmentations as augs
from patches.dataset import PatchesDataset


class LocalizationPipeline(pl.LightningModule):  # pylint: disable=R0901
    def __init__(
        self,
        train_labels: str,
        valid_labels: str,
        model: torch.nn.Module,
        resolution: tuple[int, int],
        build_optimizer,
        build_scheduler,
        loss,
        batch_size: int = 4,
    ) -> None:
        super().__init__()
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.model = model
        self.resolution = resolution
        self.loss = loss
        self.build_optimizer = build_optimizer
        self.build_scheduler = build_scheduler

        self.batch_size = batch_size

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            PatchesDataset(
                label_path=self.train_labels,
                pipeline=augs.train(self.resolution),
            ),
            batch_size=self.batch_size,
            num_workers=12,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            PatchesDataset(
                label_path=self.valid_labels,
                pipeline=augs.valid(self.resolution),
            ),
            batch_size=16,
            num_workers=12,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def configure_optimizers(self) -> tuple:
        optimizer = self.build_optimizer(
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = self.build_scheduler(
            optimizer=optimizer,
        )

        self.optimizers = [optimizer]  # type: ignore
        return self.optimizers, [scheduler]  # type: ignore

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):  # type: ignore
        images, y_true = batch
        y_pred = self.forward(images.float())
        loss = self.loss(
            y_pred,
            y_true,
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=images.shape[0],
        )

        self.log(
            "lr",
            self._get_current_lr(),
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=images.shape[0],
        )

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):  # type: ignore
        images, y_true = batch
        y_pred = self.forward(images.float())
        loss = self.loss(
            y_pred,
            y_true,
        )

        self.log(
            "valid_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=images.shape[0],
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log(
            "epoch",
            self.trainer.current_epoch,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )  # type: ignore

    def _get_current_lr(self) -> torch.Tensor:  # type: ignore
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore # noqa
        return torch.from_numpy(np.array([lr]))[0].to(self.device)
