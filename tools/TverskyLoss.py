import pytest
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from tools.DiceLoss import to_tensor

def soft_tversky_score(
    output: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)  # TP
        fp = torch.sum(output * (1.0 - target), dim=dims)
        fn = torch.sum((1 - output) * target, dim=dims)
    else:
        intersection = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1.0 - target))
        fn = torch.sum((1 - output) * target)

    tversky_score = (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth).clamp_min(eps)
    return tversky_score


class TverskyLoss(_Loss):
    def __init__(self, classes=None, log_loss=False, from_logits=True, smooth=1e-7, ignore_index=None,eps=1e-7,alpha=0.5, beta=0.5, gamma=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

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

    def forward(self,y_pred, y_true):
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

        '''
                output: torch.Tensor,
                target: torch.Tensor,
                alpha: float,
                beta: float,
                smooth: float = 0.0,
                eps: float = 1e-7,
                dims = None,
                scores = soft_tversky_score(y_pred, y_true.type_as(y_true), smooth=self.smooth, eps=self.eps, dims=dims)
        '''
        scores = soft_tversky_score(y_pred, y_true.type_as(y_true), alpha=self.alpha, beta=self.beta, )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean() ** self.gamma








