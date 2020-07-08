import torch
import torch.nn as nn

class Depth_Loss(nn.Module):
    def __init__(self):
        super(Depth_Loss, self).__init__()

    def forward(self, pred, label):
        n, c, h, w = pred.size()
        assert c == 1

        pred = pred.squeeze()
        label = label.squeeze()

        adiff = torch.abs(pred - label)
        batch_max = 0.2 * torch.max(adiff).item()
        t1_mask = adiff.le(batch_max).float()
        t2_mask = adiff.gt(batch_max).float()
        t1 = adiff * t1_mask
        t2 = (adiff * adiff + batch_max * batch_max) / (2 * batch_max)
        t2 = t2 * t2_mask
        return (torch.sum(t1) + torch.sum(t2)) / torch.numel(pred.data)