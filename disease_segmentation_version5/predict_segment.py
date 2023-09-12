import os
import shutil

import PIL.Image
import cv2
import pandas

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
from labelme import utils

import minio_util
from disease_segmentation_version5 import prediction_dia
from disease_segmentation_version5.model.seg_model import PredictSegmentModel
from disease_segmentation_version5.ssc_mask_dataset import SScPredictDataset
# import prediction_dia
from Patient_info import Patient

ckpt_path = r'.\lightning_logs\version_5\checkpoints\epoch=19-step=400.ckpt'
batch_size = 1
n_classes = 2
precision = 32
center = '省医院'
# 华西阈值：700
# 省医院阈值：1300
input_dir = r".\segment_img_input"
input_dir = os.path.join(input_dir, center)
# output_dir = r'D:\Projects\prediction\src\main\resources\static\segment_imgs'
url = r'http://127.0.0.1:8080/segment_imgs/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def init_model():
    seg = models.segmentation.deeplabv3_resnet50(num_classes=2)
    model = PredictSegmentModel(model=seg, n_class=n_classes, loss_fun=None, metrics={}, get_output=lambda z: z['out'],
                                with_info=True)

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(accelerator='gpu', devices=[0], precision=precision, logger=False, log_every_n_steps=0)

    return model, trainer


def main(model, trainer):
    patients = dict()

    dataset = SScPredictDataset(root=input_dir)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, pin_memory=True)
    prediction = trainer.predict(model, dataloader, ckpt_path=ckpt_path)

    df = pandas.DataFrame()
    df['中心'] = [center for _ in range(len(dataset))]
    df['id'] = [dataset.classes[ID] for ID in dataset.targets]
    file_names = [img[0].split("\\")[-1] for img in dataset.imgs]
    df['file_name'] = file_names
    img_nparr = predict_segment_nparr(prediction)
    df['nparr'] = [arr for arr in img_nparr]

    for idx, data in df.iterrows():
        print(idx)
        # 进行直径计算
        img, inner_points_list, outer_points_list, all_vessel_num, calc_vessel_num, max_dia, sum_dia = prediction_dia.main(
            np_arr=data["nparr"])
        filename = data["file_name"]
        patient_id = data['id']

        if patient_id not in patients.keys():
            if all_vessel_num == 0:
                patients[patient_id] = Patient(all_vessel_num, calc_vessel_num, max_dia, sum_dia, 0)
            else:
                patients[patient_id] = Patient(all_vessel_num, calc_vessel_num, max_dia, sum_dia)
        elif all_vessel_num != 0:
            patients[patient_id].update(all_vessel_num, calc_vessel_num, max_dia, sum_dia)

        # file_path = ""
        # file_path = prediction_dia.draw_save(img, inner_points_list, outer_points_list, filename)

    data = []
    for patient_id in patients.keys():
        p = patients.get(patient_id)
        if p.calc_vessel_num == 0:
            data.append([center, patient_id, 0, 0, 0, 0, 0])
        else:
            N = p.all_vessel_num / p.count
            M = p.calc_vessel_num / p.count
            D = p.max_dia
            CSURI = D * M / (N * N)
            data.append([center, patient_id, N, p.dia_sum / p.all_vessel_num, D, M, CSURI])

    print(data)
    output = pandas.DataFrame(data, columns=['中心', 'id', 'N:远端管袢数量', '平均袢顶直径', 'D:最大袢顶直径',
                                             'M:巨大管袢数量', 'CSURI:D*M/N^2'])
    output.to_excel(str(center) + "分割预测结果.xls", encoding='utf-8')
    # return file_path, file_names, input_dir + '\\0\\' + filename, all_vessel_num, calc_vessel_num, max_dia, mean_dia


def predict_segment_nparr(pred):
    prediction_narr = []
    for pre_batch in tqdm(pred):
        x, y_pred, info = pre_batch
        # filename = f'{info["filename"][0]}.jpg'
        prediction_narr.append(y_pred[0].type(torch.uint8).squeeze(0).numpy())

    return prediction_narr


if __name__ == '__main__':
    model, trainer = init_model()
    main(model, trainer)
