import os

import numpy as np

from PIL import Image
import torch
import torch.nn.functional as F

from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")
model_name = 'model_5class/best-0.833.pth'
# 不使用科学计数法

target_names = ["normal pattern", "non-specific abnormality pattern", "early pattern", "active pattern",
                "late pattern"]

def init_model():
    np.set_printoptions(suppress=True)
    print("-------------------------------------")
    print("初始化GPU...")

    # 有 GPU 就用 GPU，没有就用 CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("使用GPU:", device)
    print("初始化GPU成功！")

    print("载入模型...")
    model = torch.load(model_name)
    model = model.eval().to(device)
    print("载入模型：", model_name)
    print("载入模型成功！")
    print("-------------------------------------")
    return model, device


def one_test(img_path, model, device):

    # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                         ])

    img_pil = Image.open(img_path).convert('RGB')
    tem_name = img_path.split("/")[-1]

    input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    pred_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数

    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算

    y_score = pred_softmax[0].cpu().detach().numpy()
    target_name = target_names[np.argmax(y_score)]
    y_score = np.round(y_score, 4)
    y_score = np.array(y_score, dtype=str)
    return tem_name, y_score, target_name


def main(img_path, model, device):
    file_name, y_score, prediction_class = one_test(img_path, model, device)
    return file_name, y_score, prediction_class


if __name__ == "__main__":
    model, device = init_model()
    one_test('185.jpg', model=model, device=device)




