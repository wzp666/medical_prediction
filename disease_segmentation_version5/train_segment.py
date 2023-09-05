import os

import lightning.pytorch as pl
import torch
import torchvision.transforms as T
from lightning import seed_everything
from torch.utils.data import DataLoader
from torchvision import models

from losses import EnhanceEdgeCrossEntropyLoss2d
from model.seg_model import SegmentModel
from model.show_prediction_callback import ShowPredCallback, show_predictions

seed = seed_everything()

root_dir = '.'
resume_path = r'D:\Codes\AI\dieases_segmentation\lightning_logs\version_5\checkpoints\epoch=19-step=400.ckpt'
dataset_dir = 'D:/DataSets/AI/diease_segment_split'
batch_size = 4
learning_rate = 1e-4
n_classes = 2
precision = 16
max_epoch = 20

radius = 20

from torchmetrics import Accuracy
from torchmetrics import JaccardIndex

metrics = {
    'Accuracy': Accuracy(task="multiclass", num_classes=n_classes, average="macro", ignore_index=0),
    'global_Acc': Accuracy(task="multiclass", num_classes=n_classes, average="macro"),
    'IoU': JaccardIndex(task="multiclass", num_classes=n_classes, average="macro", ignore_index=0),
    'global_IOU': JaccardIndex(task="multiclass", num_classes=n_classes, average="macro"),
}
seg = models.segmentation.deeplabv3_resnet50(num_classes=2)

model = SegmentModel(model=seg,
                     n_class=n_classes,
                     loss_fun=EnhanceEdgeCrossEntropyLoss2d(radius=20),
                     metrics=metrics,
                     lr=learning_rate,
                     batch_size=batch_size,
                     precision=precision,
                     seed=seed
                     )

transform = None
# T.Compose([
#     T.Normalize(
#         mean=[0.5, 0.5, 0.5],
#         std=[0.5, 0.5, 0.5])
# ])

from ssc_mask_dataset import SScDataset

train_dataset = SScDataset(root_dir=dataset_dir, mode='train', transform=transform)
val_dataset = SScDataset(root_dir=dataset_dir, mode='val', transform=transform)

train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=batch_size, shuffle=False, pin_memory=True)

torch.set_float32_matmul_precision('medium')

trainer = pl.Trainer(accelerator='gpu', devices=[0], precision=precision,
                     callbacks=[
                         ShowPredCallback(train_dataset, label='train', sub_dir='predictions'),
                         ShowPredCallback(val_dataset, label='val', sub_dir='predictions')
                     ],
                     max_epochs=max_epoch, log_every_n_steps=1,
                     )


def predict_all():
    path = './all_predictions/'
    os.makedirs(path, exist_ok=True)

    predictions = trainer.predict(model, [train_dataloader, val_dataloader], ckpt_path=resume_path)
    train_pred = predictions[0]
    val_pred = predictions[1]

    to_pillow = T.ToPILImage()

    def save_img(pred, label):
        counter = 0
        for pre_batch in pred:
            x, y, y_pred, scores = pre_batch
            for i in range(x.shape[0]):
                score_str = ''
                for name, score in scores.items():
                    score_str += f'-{name}({score:.3f})'

                filename = f'{label}-{counter}-{score_str}.png'
                show_predictions(
                    [to_pillow(x[i]), to_pillow(y[i].type(torch.float)), to_pillow(y_pred[i].type(torch.float))],
                    ['Original Image', 'Ground Truth', 'Prediction'],
                    os.path.join(path, filename), plt_show=False)
                counter += 1

    save_img(train_pred, 'train')
    save_img(val_pred, 'val')


if __name__ == '__main__':
    # Train!
    # trainer.fit(model, train_dataloader, val_dataloader,
    #             # ckpt_path=resume_path
    #             )
    # Eval!
    predict_all()
