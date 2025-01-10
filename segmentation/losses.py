import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import FocalLoss

def one_hot_encoding(targets, C=2):
    return F.one_hot(targets.long().squeeze(1), num_classes=C).permute(0, 4, 1, 2, 3).float()

class L1DFL(nn.Module):
    def __init__(self, gamma=2, bins=10, epsilon=0.1):
        super(L1DFL, self).__init__()
        self.gamma = gamma
        self.bins = bins
        self.epsilon = epsilon
        self.edges = torch.linspace(0, 1, bins + 1).cuda()
        self.focal = FocalLoss(gamma=self.gamma, use_softmax=True, to_onehot_y=False)

    def forward(self, logits, labels):
        logits = logits.type(torch.float64)
        labels = labels.type(torch.float64)
        probabilities = torch.softmax(logits, 1, torch.float64)
        targets = one_hot_encoding(labels)
        focal_loss = self.focal(logits, targets)
        
        gradients = torch.abs(probabilities - targets)
        g = gradients.view(-1)
        N = g.size(0)
        gd = torch.zeros_like(g)
        # counts = []
        for center in self.edges:
            mask = (g >= center - self.epsilon/2) & (g < center + self.epsilon/2)
            count_in_region = mask.sum().item()
            valid_length = min(center + self.epsilon/2, 1) - max(center - self.epsilon/2, 0)
            gd[mask] = (count_in_region / valid_length).type(torch.float64)
            # counts.append(count_in_region / valid_length)
        beta = N / (gd + 1e-8)
        beta = beta.view_as(gradients)
        weighted_intersection = (beta * probabilities * targets).sum(dim=[2, 3, 4])
        weighted_union = (beta*(probabilities**2 + targets**2)).sum(dim=[2, 3, 4])
        dice_score = (2. * weighted_intersection + 1e-8) / (weighted_union + 1e-8)
        loss = (1 - dice_score.mean())+focal_loss
        return loss