import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def onehot(mask: torch.Tensor):
    shape = list(mask.shape)
    shape[1] = 2
    mask = mask.view(-1)
    mask_onehot = F.one_hot(mask, num_classes=2)
    mask_onehot = mask_onehot.transpose(0, 1).reshape(shape)
    return mask_onehot


def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    # cls_w = class_weight.compute_claszs_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


from model.mask_enhance import masks_enhace_edge


class EnhanceEdgeCrossEntropyLoss2d(nn.Module):
    def __init__(self, adaptive_weight=True, weight=None,
                 radius: int = None):
        """
        :param weight: None to adaptive weight
        :param enhance_ratio: loss = (1-enhance_ratio)*global_loss + enhance_ratio*enhance_loss
        :param enhance_ignore_index: enhance part None for all enhance
        """
        super(EnhanceEdgeCrossEntropyLoss2d, self).__init__()
        self.adaptive_weight = adaptive_weight
        self.radius = radius
        self.CE = nn.CrossEntropyLoss(weight=weight)
        self.EnhancedCE = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, output, target, with_batch=True):
        ch_idx = 1 if with_batch else 0
        if len(target.shape) == len(output.shape):
            assert target.shape[ch_idx] == 1, 'should be 1 in channel dimension'
            target = target.squeeze(ch_idx)

        enhance_mask = masks_enhace_edge(target, radius=self.radius)

        if self.adaptive_weight:
            adaptive_ratio = torch.sum(target) / target.numel()
            loss = F.cross_entropy(output, target,
                                   weight=torch.asarray([adaptive_ratio, 1 - adaptive_ratio], device=output.device))
        else:
            loss = self.CE(output, target)

        masked_output = output * enhance_mask + -1 * (1 - enhance_mask)
        masked_target = target * enhance_mask.squeeze(ch_idx) + -1 * (1 - enhance_mask.squeeze(ch_idx))
        enhance_loss = self.EnhancedCE(masked_output, masked_target)
        return loss + enhance_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss
