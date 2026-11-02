import torch
import torch.nn as nn
import torch.nn.functional as F

# 不影响指标计算
# class SoftLoULoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super(SoftLoULoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, pred, target):
#         # pred: logits
#         pred = torch.sigmoid(torch.clamp(pred, -50, 50))  # ✅ 稳定 sigmoid
#         smooth = self.smooth

#         intersection = pred * target
#         intersection_sum = torch.sum(intersection, dim=(1,2,3))
#         pred_sum = torch.sum(pred, dim=(1,2,3))
#         target_sum = torch.sum(target, dim=(1,2,3))

#         iou = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)
#         loss = 1 - torch.mean(iou)
#         return loss

# 解决数值不稳定loss 但是影响到了指标计算
# class SoftLoULoss(nn.Module):
#     def __init__(self, smooth: float = 1e-6, clamp: float = 50.0):
#         super().__init__()
#         self.smooth = smooth
#         self.clamp = clamp

#     def forward(self, pred, target):
#         # pred: logits
#         pred = torch.clamp(pred, -self.clamp, self.clamp)
#         pred = torch.sigmoid(pred)

#         smooth = self.smooth
#         intersection = pred * target

#         intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
#         pred_sum = torch.sum(pred, dim=(1, 2, 3))
#         target_sum = torch.sum(target, dim=(1, 2, 3))

#         iou = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)
#         loss = 1 - torch.mean(iou)
#         return loss

# 原来loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLoULoss(nn.Module):
    def __init__(self):
        super(SoftLoULoss, self).__init__()

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        smooth = 1

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)

        return loss