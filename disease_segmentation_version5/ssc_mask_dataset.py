import glob
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from typing import Any, Tuple

from torchvision.datasets import ImageFolder
from torchvision.io import read_image

from pathlib import Path


class SScDataset(Dataset):
    image_dir_name = 'images'
    mask_dir_name = 'masks'

    def __init__(self, *,
                 root_dir,
                 mode='train',
                 transform=None,
                 target_transform=None):
        super().__init__()
        self.loader = read_image
        self.transform = transform
        self.target_transform = target_transform
        assert mode == 'train' or mode == 'val'
        assert os.path.exists(os.path.join(root_dir, self.image_dir_name))
        assert os.path.exists(os.path.join(root_dir, self.mask_dir_name))
        image_paths = glob.glob(os.path.join(root_dir, self.image_dir_name, mode, '*.jpg'))
        mask_paths = glob.glob(os.path.join(root_dir, self.mask_dir_name, mode, '*.png'))
        assert len(image_paths) == len(mask_paths) and len(image_paths) != 0
        self.samples = []
        for i in range(len(image_paths)):
            self.samples.append((image_paths[i], mask_paths[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, mask_path = self.samples[index]
        img = self.loader(img_path)
        mask = self.loader(mask_path)

        if torch.rand(1).item() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # if torch.rand(1).item() > 0.5:
        #     img = TF.vflip(img)
        #     mask = TF.vflip(mask)

        img = img.type(torch.float)
        img = img / 255.
        mask = mask.type(torch.int64)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask


class SScPredictDataset(ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, _ = self.samples[index]
        sample = self.loader(path)

        # image filename
        return TF.to_tensor(sample), {'sub_dir': os.path.basename(os.path.dirname(path)), 'filename': Path(path).stem}
