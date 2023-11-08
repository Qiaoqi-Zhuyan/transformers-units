import pytest
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import numpy as np
import einops
'''
Dice Loss 常用于语义分割领域, 计算分割结果与真实结果重叠区域
'''


# modify from https://github.com/Beckschen/TransUNet/blob/main/utils.py#L9
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


# modify from https://github.com/Qiaoqi-Zhuyan/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/dice.py#L12
def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score


def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x


class MultiClassDiceLoss(_Loss):
    def __init__(self, classes=None, log_loss=False, from_logits=True, smooth=1e-5, ignore_index=None, eps=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        if classes is not None:
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.smooth = smooth
        self.from_logits = from_logits
        if self.from_logits:
            self.log_softmax = F.log_softmax(dim=1)
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        '''

        :param y_pred:  tensor([B, C, H, W])
        :param y_true:  tensor([B, H, W]) or (B, C, H W)
        :return:
        '''

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            y_pred = torch.exp(self.log_softmax(y_pred))

        batch_size = y_true.size(0)
        num_classes = y_pred.size(1)
        dims=(0, 2)

        y_true = y_true.view(batch_size, -1)

        y_pred = y_pred.view(batch_size, num_classes, -1)

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask.unsqueeze(1)

            y_true = F.one_hot((y_true * mask).to(torch.long), num_classes) # [N,H*W]-> [N,H*W,C]
            y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1) #[N, C, H*W]
        else:
            y_true = F.one_hot(y_true, num_classes) #[N,H*W] -> [N,H*W,C]
            y_true = y_true.permute(0, 2, 1) #[N, C, H*W]

        #def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        #    return soft_dice_score(output, target, smooth, eps, dims)

        scores = soft_dice_score(y_pred, y_true.type_as(y_true), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()


class BinaryClassDiceLoss(_Loss):
    def __init__(self, classes=None, log_loss=False, from_logits=True, smooth=1e-5, ignore_index=None, eps=1e-6):
        super(BinaryClassDiceLoss, self).__init__()
        if classes is not None:
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.smooth = smooth
        self.from_logits = from_logits
        if self.from_logits:
            self.log_sigmoid = F.logsigmoid()
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        '''

        :param y_pred:  tensor([B, C, H, W])
        :param y_true:  tensor([B, H, W]) or (B, C, H W)
        :return:
        '''

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            y_pred = torch.exp(self.log_sigmoid(y_pred))

        batch_size = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(batch_size, 1, -1)

        y_pred = y_pred.view(batch_size, 1, -1)

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask
            y_true = y_true * mask

        # def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        #    return soft_dice_score(output, target, smooth, eps, dims)

        scores = soft_dice_score(y_pred, y_true.type_as(y_true), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()


class MultiLabelDiceLoss(_Loss):
    def __init__(self, classes=None, log_loss=False, from_logits=True, smooth=1e-5, ignore_index=None, eps=1e-6):
        super(MultiLabelDiceLoss, self).__init__()
        if classes is not None:
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.smooth = smooth
        self.from_logits = from_logits
        if self.from_logits:
            self.log_sigmoid = F.logsigmoid()
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        '''

        :param y_pred:  tensor([B, C, H, W])
        :param y_true:  tensor([B, H, W]) or (B, C, H W)
        :return:
        '''

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            y_pred = torch.exp(self.log_sigmoid(y_pred))

        batch_size = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(batch_size, num_classes, -1)

        y_pred = y_pred.view(batch_size, num_classes, -1)

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask
            y_true = y_true * mask

        # def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        #    return soft_dice_score(output, target, smooth, eps, dims)

        scores = soft_dice_score(y_pred, y_true.type_as(y_true), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()


