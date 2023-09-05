from typing import Any

import torch
from torch import optim, nn
import pytorch_lightning as pl
import torch.nn.functional as F


class PredictSegmentModel(pl.LightningModule):
    def __init__(self, model, n_class, loss_fun, metrics, with_truth=False, with_info=False, *args, **kwargs):
        super().__init__()
        self.model = model
        self.n_class = n_class
        self.loss_fun = loss_fun
        self.metrics = metrics
        assert not (with_truth and with_info), 'can not be both truth and info'
        self.with_truth = with_truth
        self.with_info = with_info

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if self.with_truth:
            x, y = batch
            z = self.model(x)
            # no softmax when predict
            y_hat = z['out']
            y_pred = torch.argmax(y_hat, dim=1, keepdim=True)

            scores = {}
            for name, metric in self.metrics.items():
                metric = metric.to(y_pred.device)
                value = metric(y_pred.float(), y)
                scores[name] = value

            return x, y, y_pred, scores
        elif self.with_info:
            x, info = batch
            z = self.model(x)
            y_hat = z['out']
            y_pred = torch.argmax(y_hat, dim=1, keepdim=True)

            return x, y_pred, info
        else:
            x = batch
            z = self.model(x)
            y_hat = z['out']
            y_pred = torch.argmax(y_hat, dim=1, keepdim=True)

            return x, y_pred


class SegmentModel(pl.LightningModule):
    def __init__(self, model, n_class, loss_fun, metrics, lr=0.001, *args, **kwargs):
        super().__init__()
        self.model = model
        self.n_class = n_class
        self.loss_fun = loss_fun
        self.metrics = metrics
        self.lr = lr
        self.training_step_outputs = {'train/loss': []}
        for name, metric in self.metrics.items():
            self.training_step_outputs[f"train/{name}"] = []
        self.validation_step_outputs = {'val/loss': []}
        for name, metric in self.metrics.items():
            self.validation_step_outputs[f"val/{name}"] = []
        self.save_hyperparameters(ignore=['model', 'loss_fun', 'metrics'])

    def forward(self, batch: Any):
        batch = batch.to('cuda')
        x = batch
        z = self.model(x)
        y_hat = torch.softmax(z['out'], dim=1)
        y_pred = torch.argmax(y_hat, dim=1, keepdim=True)
        return y_pred

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        z = self.model(x)
        # no softmax when predict
        y_hat = z['out']
        y_pred = torch.argmax(y_hat, dim=1, keepdim=True)

        scores = {}
        for name, metric in self.metrics.items():
            metric = metric.to(y_pred.device)
            value = metric(y_pred.float(), y)
            scores[name] = value

        return x, y, y_pred, scores

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        z = self.model(x)
        y_hat = torch.softmax(z['out'], dim=1)

        loss = self.loss_fun(y_hat, y)

        self.log("train/step_loss", loss, prog_bar=True)
        self.training_step_outputs["train/loss"].append(loss)

        y_pred = torch.argmax(y_hat, dim=1, keepdim=True)

        for name, metric in self.metrics.items():
            metric = metric.to(y_pred.device)
            value = metric(y_pred.float(), y)
            self.log(f"train/{name}", value)
            self.training_step_outputs[f"train/{name}"].append(value)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        y_hat = torch.softmax(z['out'], dim=1)

        loss = self.loss_fun(y_hat, y)
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
