import os

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as T

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


def show_predictions(display_list, titles, filename=None, plt_show=True):
    assert len(display_list) == len(titles)
    num = len(display_list)
    plt.figure(figsize=(num * 4, 4))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.title(titles[i])
        plt.imshow(display_list[i])
    if filename is not None:
        plt.savefig(filename)
    if plt_show:
        plt.show()
    plt.close()


class ShowPredCallback(Callback):
    def __init__(self, dataset, label, sub_dir='./predictions/') -> None:
        super().__init__()
        self.dataset = dataset
        self.label = label
        self.sub_dir = sub_dir

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        sample, mask = self.dataset[0]

        pl_module.eval()
        y_pred = pl_module(sample[None, ...])[0]

        to_pillow = T.ToPILImage()

        dir = os.path.join(trainer.log_dir, self.sub_dir)

        os.makedirs(dir, exist_ok=True)

        show_predictions([to_pillow(sample), to_pillow(mask.type(torch.float)), to_pillow(y_pred.type(torch.float))],
                         ['image', 'mask', 'prediction'],
                         os.path.join(dir, f'epoch_{self.label}_{trainer.global_step}.png'))
