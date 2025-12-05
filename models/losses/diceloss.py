import torch
import torch.nn.functional as F
from torch import nn

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = torch.sum(pred * target)
    pred_sum = torch.sum(pred * pred)
    target_sum = torch.sum(target * target)

    return 1 - ((2. * intersection + 1e-5) / (pred_sum + target_sum + 1e-5))


def dice_loss_wrong(predictions, targets, smooth=1e-6):
    """
    Calculate the Dice loss.

    Args:
        predictions (torch.Tensor): Predicted segmentation maps (shape: N, C, H, W).
        targets (torch.Tensor): Ground truth segmentation maps (shape: N, C, H, W).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Calculated Dice loss.
    """
    # Flatten the tensors
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Calculate intersection and union
    intersection = (predictions * targets).sum()
    dice = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)

    # Dice loss is 1 - Dice coefficient
    return 1 - dice


# Пример использования
# num_classes = 3
# pred = torch.randn(4, num_classes, 256, 256)  # предсказания модели
# target = torch.randint(0, num_classes, (4, 256, 256))  # истинные метки

# loss = multiclass_dice_loss(pred, target, num_classes)
# print(loss)

