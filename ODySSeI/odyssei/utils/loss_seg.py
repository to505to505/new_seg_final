import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.soft_skeleton import SoftSkeletonize


def dice_loss(output_net, target_net):
    """
    Dice Loss Computation

    Arguments:
        output_net (torch.tensor): Batch of model outputs
        target_net (torch.tensor): Batch of corresponding ground truths

    Returns:
        Dice loss
    """

    pred_net = torch.sigmoid(output_net)

    smooth = 1

    intersection = (pred_net * target_net).sum()

    dice_coeff = (2.0 * intersection + smooth) / (
        pred_net.sum() + target_net.sum() + smooth
    )

    return 1 - dice_coeff


def iou_loss(output_net, target_net):
    """
    IoU Loss Computation

    Arguments:
        output_net (torch.tensor): Batch of model outputs
        target_net (torch.tensor): Batch of corresponding ground truths

    Returns:
        IoU loss
    """

    pred_net = torch.sigmoid(output_net)

    smooth = 1

    intersection = (pred_net * target_net).sum()

    total = (pred_net + target_net).sum()

    iou = (intersection + smooth) / (total - intersection + smooth)

    return 1 - iou


class dice_with_bce_loss(nn.Module):

    """
    Dice with Weighted BCE Loss Computation

    Attributes:
        bce_pos_w (int): Weightage given to positive samples in BCE loss (addresses class imbalance)
        alpha (float): Overall weightage given to BCE loss
    """

    def __init__(self, bce_pos_w=2, alpha=0.5):
        super(dice_with_bce_loss, self).__init__()
        self.bce_pos_w = bce_pos_w
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                self.bce_pos_w,
            )
        )

    def forward(self, output_net, target_net):
        return (1 - self.alpha) * dice_loss(
            output_net, target_net
        ) + self.alpha * self.bce_loss(output_net, target_net)


class soft_cldice(nn.Module):

    """
    Adapted from the official repo: https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/cldice.py
    """

    def __init__(self, iter_=10, smooth=1.0, exclude_background=False):
        super(soft_cldice, self).__init__()
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=iter_)
        self.exclude_background = exclude_background

    def forward(self, y_out, y_true):
        y_pred = torch.sigmoid(y_out)

        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]

        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / (
            torch.sum(skel_pred) + self.smooth
        )
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / (
            torch.sum(skel_true) + self.smooth
        )
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)

        return cl_dice


class soft_dice_cldice(nn.Module):

    """
    Adapted from the official repo: https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/cldice.py
    """

    def __init__(self, iter_=10, alpha=0.5, smooth=1.0, exclude_background=False):
        super(soft_dice_cldice, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.soft_skeletonize = SoftSkeletonize(num_iter=iter_)
        self.exclude_background = exclude_background

    def forward(self, y_out, y_true):
        y_pred = torch.sigmoid(y_out)

        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]

        dice = dice_loss(y_out, y_true)
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / (
            torch.sum(skel_pred) + self.smooth
        )
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / (
            torch.sum(skel_true) + self.smooth
        )
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)

        return (1.0 - self.alpha) * dice + self.alpha * cl_dice
