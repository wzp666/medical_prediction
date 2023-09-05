import os
import shutil

import PIL.Image
import cv2

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.utils import save_image
from tqdm import tqdm

from labelme import utils

import minio_util
from disease_segmentation_version5 import prediction_dia
from disease_segmentation_version5.model.seg_model import PredictSegmentModel
from disease_segmentation_version5.ssc_mask_dataset import SScPredictDataset
# import prediction_dia

ckpt_path = r'D:\Projects\medical_prediction_py_service\model_segment\epoch=19-step=400.ckpt'
batch_size = 1
n_classes = 2
precision = 32
input_dir = r"D:\Projects\medical_prediction_py_service\disease_segmentation_version5\segment_img_input"
output_dir = r'D:\Projects\prediction\src\main\resources\static\segment_imgs'
url = r'http://127.0.0.1:8080/segment_imgs/'
# 'http://127.0.0.1:8080/segment_imgs/1_mask.png'


def init_model():
    seg = models.segmentation.deeplabv3_resnet50(num_classes=2)
    model = PredictSegmentModel(model=seg, n_class=n_classes, loss_fun=None, metrics={}, with_info=True)

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(accelerator='gpu', devices=[0], precision=precision, logger=False, log_every_n_steps=0)

    return model, trainer


def main(model, trainer):

    dataset = SScPredictDataset(root=input_dir)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, pin_memory=True)
    prediction = trainer.predict(model, dataloader, ckpt_path=ckpt_path)

    img_nparr, filename = predict_segment_nparr(prediction)


    # 进行直径计算
    img, inner_points_list, outer_points_list, all_vessel_num, calc_vessel_num, max_dia, mean_dia = prediction_dia.main(np_arr=img_nparr)
    file_path = prediction_dia.draw_save(img, inner_points_list, outer_points_list, filename)
    return file_path, filename, input_dir + '\\0\\' + filename, all_vessel_num, calc_vessel_num, max_dia, mean_dia


def predict_segment_nparr(pred):
    for pre_batch in tqdm(pred):
        x, y_pred, info = pre_batch
        filename = f'{info["filename"][0]}.jpg'
        prediction01 = y_pred[0].type(torch.uint8).squeeze(0).numpy()
        return prediction01, filename


if __name__ == '__main__':

    model, trainer = init_model()
    main(model, trainer)

