from typing import Any

import torch
from torch import optim, nn
import lightning.pytorch as pl
import torch.nn.functional as F

from model.mask_enhance import masks_enhace_edge


def onehot(mask: torch.Tensor):
    shape = list(mask.shape)
    shape[1] = 2
    mask = mask.view(-1)
    mask_onehot = F.one_hot(mask, num_classes=2)
    mask_onehot = mask_onehot.transpose(0, 1).reshape(shape)
    return mask_onehot


class SegmentModelAdaptLoss(pl.LightningModule):
    def __init__(self, model, n_class, loss_fun, extra_loss_fun, metrics, lr):
        super().__init__()
        self.model = model
        self.n_class = n_class
        self.loss_fun = loss_fun
        self.extra_loss_fun = extra_loss_fun
        self.metrics = metrics
        self.lr = lr
        self.training_step_outputs = {'train/loss': []}
        for name, metric in self.metrics.items():
            self.training_step_outputs[f"train/{name}"] = []
        self.validation_step_outputs = {'val/loss': []}
        for name, metric in self.metrics.items():
            self.validation_step_outputs[f"val/{name}"] = []

    def forward(self, batch: Any):
        batch = batch.to('cuda')
        x = batch
        z = self.model(x)
        y_hat = torch.softmax(z['out'], dim=1)
        y_pred = torch.argmax(y_hat, dim=1, keepdim=True)
        return y_pred

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, _ = batch
        z = self.model(x)
        # no softmax when predict
        y_hat = z['out']
        y_pred = torch.argmax(y_hat, dim=1, keepdim=True)
        return y_pred

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        z = self.model(x)
        y_hat = torch.softmax(z['out'], dim=1)

        ratio = torch.sum(y) / y.numel()

        loss = F.cross_entropy(y_hat, y.squeeze(1), weight=torch.asarray([ratio, 1 - ratio], device=y_hat.device))

        enhance = masks_enhace_edge(y.squeeze(1), 10)
        extra_loss = self.extra_loss_fun(y_hat * enhance, (y * enhance).squeeze(1))

        self.log("train/step_loss", loss, prog_bar=True)
        self.training_step_outputs["train/loss"].append(loss)

        y_pred = torch.argmax(y_hat, dim=1, keepdim=True)

        for name, metric in self.metrics.items():
            metric = metric.to(y_pred.device)
            value = metric(y_pred.float(), y)
            self.log(f"train/{name}", value)
            self.training_step_outputs[f"train/{name}"].append(value)

        return 0.5 * loss + 0.5 * extra_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        y_hat = torch.softmax(z['out'], dim=1)

        ratio = torch.sum(y) / y.numel()

        loss = F.cross_entropy(y_hat, y.squeeze(1), weight=torch.asarray([ratio, 1 - ratio], device=y_hat.device))

        self.validation_step_outputs["val/loss"].append(loss)

        y_pred = torch.argmax(y_hat, dim=1, keepdim=True)

        for name, metric in self.metrics.items():
            metric = metric.to(y_pred.device)
            value = metric(y_pred.float(), y)
            self.validation_step_outputs[f"val/{name}"].append(value)

        return loss

    def on_train_epoch_end(self):
        for name, scores in self.training_step_outputs.items():
            mean_score = torch.stack(scores).mean()
            print(f'{name}: {mean_score}')
            self.log(name, mean_score, sync_dist=True)
            scores.clear()

    def on_validation_epoch_end(self):
        for name, scores in self.validation_step_outputs.items():
            mean_score = torch.stack(scores).mean()
            self.log(name, mean_score, sync_dist=True)
            print(f'{name}: {mean_score}')
            scores.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
