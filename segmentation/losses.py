import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import FocalLoss

def one_hot_encoding(targets, C=2):
    return F.one_hot(targets.long().squeeze(1), num_classes=C).permute(0, 4, 1, 2, 3).float()

class L1DFL(nn.Module):
    def __init__(self, gamma=2, epsilon=0.05):
        super().__init__()
        self.gamma = gamma
        self.bins = int((1/epsilon) +1)
        self.epsilon = epsilon
        self.register_buffer("bin_centers", torch.linspace(0, 1, self.bins))
        self.focal = FocalLoss(gamma=self.gamma, use_softmax=True, to_onehot_y=False)

    def forward(self, logits, labels):
        device = logits.device

        if logits.ndim == 5:  # 3D
            spatial_dims = [2, 3, 4]
        elif logits.ndim == 4:  # 2D
            spatial_dims = [2, 3]
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
        
        logits = logits.float()
        probabilities = torch.softmax(logits, dim=1)
        
        targets = one_hot_encoding(labels)
        focal_loss = self.focal(logits, targets)

        gradients = torch.abs(probabilities - targets)
        g = gradients.view(-1)
        N = g.numel()

        bin_centers = self.bin_centers.to(device)
        left_edges = torch.clamp(bin_centers - self.epsilon / 2, min=0)
        right_edges = torch.clamp(bin_centers + self.epsilon / 2, max=1)
        widths = right_edges - left_edges

        bucket_idx = torch.bucketize(g, boundaries=left_edges, right=True) - 1
        bucket_idx = torch.clamp(bucket_idx, min=0, max=self.bins - 1)

        counts = torch.zeros_like(bin_centers, device=device)
        counts = counts.scatter_add(0, bucket_idx, torch.ones_like(g))

        density = counts / (widths + 1e-8)

        beta = density[bucket_idx]
        beta = N / (beta + 1e-8)
        beta = beta.view_as(gradients)

        weighted_intersection = (beta * probabilities * targets).sum(dim=spatial_dims)
        weighted_union = (beta * (probabilities**2 + targets**2)).sum(dim=spatial_dims)
        dice_score = (2. * weighted_intersection + 1e-8) / (weighted_union + 1e-8)

        loss = (1 - dice_score.mean()) + focal_loss
        return loss
