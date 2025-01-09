import warnings
from typing import Callable, Optional, Union
from enum import Enum
import cc3d
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss
from monai.losses import FocalLoss, DiceLoss

from monai.networks import one_hot
from monai.utils import LossReduction



class DiceFocalLoss(_Loss):
    """
    Compute both Dice loss and Focal Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Focal Loss is shown in ``monai.losses.FocalLoss``.

    ``gamma`` and ``lambda_focal`` are only used for the focal loss.
    ``include_background``, ``weight`` and ``reduction`` are used for both losses
    and other parameters are only used for dice loss.

    """
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act:None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        gamma: float = 2.0,
        focal_weight: None = None,
        weight: None = None,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
    ):
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``.
                for example: `other_act = torch.tanh`. only used by the `DiceLoss`, not for `FocalLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            gamma: value of the exponent gamma in the definition of the Focal loss.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes).
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_focal: the trade-off weight value for focal loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        weight = focal_weight if focal_weight is not None else weight
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            weight=weight,
        )
        self.focal = FocalLoss(
            include_background=include_background, to_onehot_y=False, gamma=gamma, weight=weight, reduction=reduction
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.to_onehot_y = to_onehot_y

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: the shape should be BNH[WD]. The input should be the original logits
                due to the restriction of ``monai.losses.FocalLoss``.
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )
        if self.to_onehot_y:
            n_pred_ch = input.shape[1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
        dice_loss = self.dice(input, target)
        focal_loss = self.focal(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
        return total_loss
    
class LossReduction(Enum):
    MEAN = "mean"
    SUM = "sum"
    NONE = "none"  
      
class wDiceLoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ):
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        self.register_buffer("class_weight", torch.ones(1))
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]
        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
        weight_tensor = self._calculate_weight(target)
        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis
        intersection = torch.sum(target * input * weight_tensor, dim=reduce_axis)
        if self.squared_pred:
            ground_o = torch.sum(target**2 * weight_tensor, dim=reduce_axis)
            pred_o = torch.sum(input**2 * weight_tensor, dim=reduce_axis)
        else:
            ground_o = torch.sum(target * weight_tensor, dim=reduce_axis)
            pred_o = torch.sum(input * weight_tensor, dim=reduce_axis)
        denominator = ground_o + pred_o
        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)
        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return f
    def _calculate_weight(self, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the weight tensor based on the lesion size in the target mask.

        Args:
            target: Target tensor with shape [batch_size, num_classes, height, width, depth].

        Returns:
            torch.Tensor: Weight tensor with the same shape as target.
        """
        batch_size, num_classes = target.shape[:2]
        weight_tensors = []

        for b in range(batch_size):
            class_weight_tensors = []

            for c in range(num_classes):
                target_np = target[b, c].cpu().detach().numpy()  
                labeled_array, num_features = cc3d.connected_components(target_np, connectivity=18, return_N = True) 

                weight_tensor = torch.zeros_like(target[b, c])

                for label in range(1, num_features + 1):
                    component_size = (labeled_array == label).sum()
                    weight = labeled_array.size / ((num_features + 1) * component_size)
                    weight_tensor[labeled_array == label] = weight

                class_weight_tensors.append(weight_tensor)

            weight_tensors.append(torch.stack(class_weight_tensors, dim=0))

        return torch.stack(weight_tensors, dim=0)

def one_hot_encoding(targets, C=2):
    return F.one_hot(targets.long().squeeze(1), num_classes=C).permute(0, 4, 1, 2, 3).float()

def diceLoss(inputs, targets):
    smooth = 1e-5
    inputs_flat = inputs.view(inputs.shape[0], inputs.shape[1], -1)
    targets_flat = targets.view(targets.shape[0], targets.shape[1], -1)
    intersection = (inputs_flat * targets_flat).sum(-1)
    union = inputs_flat.sum(-1) + targets_flat.sum(-1)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


class VolumePreservingLoss(nn.Module):
    def __init__(self, lambd=3e-6, c=15.0, epsilon=1e-5):
        super().__init__()
        self.lambd = lambd
        self.c = torch.tensor(c)
        self.epsilon = epsilon
        self.dicefocal = DiceFocalLoss(to_onehot_y=True, softmax=True, gamma = 2)
    def forward(self, S, Y):
        S = F.softmax(S, dim=1)
        # dice_loss = diceLoss(S, Y)
        dfl = self.dicefocal(S, Y)
        Y = one_hot_encoding(Y)
        S_tilde = (1 / (1 + torch.exp(-(S - 0.5) * self.c)) - 0.5) * \
                  (1 + torch.exp(-0.5 * self.c)) / (1 - torch.exp(-0.5 * self.c)) + 0.5
        F_S_tilde_list = []
        for class_index in range(S.shape[1]):
            sum_y = Y[:, class_index].sum()
            mask_positive = S_tilde[:, class_index] > 0.5
            false_negative = torch.where(~mask_positive, Y[:, class_index], torch.tensor(0.0, device=S.device)) * S_tilde[:, class_index]
            false_positive = torch.where(mask_positive, 1.0 - Y[:, class_index], torch.tensor(0.0, device=S.device)) * S_tilde[:, class_index]
            penalty = false_positive.sum() + false_negative.sum() + self.epsilon
            F_S_tilde_class = penalty / (sum_y + self.epsilon)
            F_S_tilde_list.append(F_S_tilde_class)
        F_S_tilde = torch.stack(F_S_tilde_list).mean()
        # total_loss = dfl + self.lambd * F_S_tilde
        return F_S_tilde
    
import torch
import torch.nn as nn

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, eps=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, true):
        # Ensure the tensors are of type float
        y_pred = F.softmax(pred, dim=1)
        y_true = one_hot_encoding(true)

        # Compute true positives, false positives, and false negatives
        true_pos = torch.sum(y_true * y_pred)
        false_neg = torch.sum(y_true * (1 - y_pred))
        false_pos = torch.sum((1 - y_true) * y_pred)

        # Compute Tversky index
        tversky_index = (true_pos + self.eps) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.eps)

        # Compute Focal Tversky loss
        focal_tversky_loss = torch.pow((1 - tversky_index), self.gamma)

        return focal_tversky_loss



def tverskyLoss(inputs, targets, alpha, beta, smooth):
    inputs_flat = inputs.view(inputs.shape[0], inputs.shape[1], -1)
    targets_flat = targets.view(targets.shape[0], targets.shape[1], -1)
    true_positives = (inputs_flat * targets_flat).sum(-1)
    false_negatives = ((1 - inputs_flat) * targets_flat).sum(-1)
    false_positives = (inputs_flat * (1 - targets_flat)).sum(-1)
    tversky_index = (true_positives + smooth) / (true_positives + alpha * false_positives + beta * false_negatives + smooth)
    return 1 - tversky_index.mean()

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

class GeneralizedDiceLoss(torch.nn.Module):
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()
        self.focal = FocalLoss(
            include_background=True, to_onehot_y=True, gamma=2.0, reduction='mean'
        )

    def forward(self, inputs, targets, smooth=1e-6):
        # Ensure softmax is applied to channel dimension
        inputs = F.softmax(inputs, dim=1)
        focalloss = self.focal(inputs, targets)
        # Convert targets to one-hot format, dimensions [N, C, D, H, W]
        targets_one_hot = one_hot_encoding(targets)  # Reduce channel dimension first if necessary
        # targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 4, 1, 2, 3).float()

        # Calculate weights for each class
        weights = 1.0 / (torch.sum(targets_one_hot, dim=(0, 2, 3, 4)) ** 2 + smooth)

        # Calculate intersection and union per class
        intersection = torch.sum(inputs * targets_one_hot, dim=(0, 2, 3, 4))
        union = torch.sum(inputs + targets_one_hot, dim=(0, 2, 3, 4))

        # Calculate weighted Dice score
        dice_score = 2.0 * torch.sum(weights * intersection) / torch.sum(weights * (union + smooth))
        loss = focalloss + (1 - dice_score)
        return loss



class VolumePreservingLossWithTversky(nn.Module):
    def __init__(self, lambd=0.02, r=0.9, c=15.0, alpha=0.7, beta=0.3, epsilon=1e-5):
        super().__init__()
        self.lambd = lambd
        self.r = r
        self.c = torch.tensor(c)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
    def forward(self, S, Y):
        S = F.softmax(S, dim=1)
        Y = one_hot_encoding(Y)
        tversky_loss = tverskyLoss(S, Y, alpha=self.alpha, beta=self.beta, smooth = self.epsilon)
        S_tilde = (1 / (1 + torch.exp(-(S - 0.5) * self.c)) - 0.5) * \
                  (1 + torch.exp(-0.5 * self.c)) / (1 - torch.exp(-0.5 * self.c)) + 0.5
        F_S_tilde_list = []
        for class_index in range(S.shape[1]):
            sum_y = Y[:, class_index].sum() + self.epsilon
            intersection = (S_tilde[:, class_index] > 0.5).float() * (Y[:, class_index] > 0.5).float()
            sum_intersection = intersection.sum()
            F_S_tilde_class = 100 * torch.abs(sum_intersection - sum_y) / sum_y
            F_S_tilde_list.append(F_S_tilde_class)
        F_S_tilde = torch.stack(F_S_tilde_list).mean()
        total_loss = tversky_loss + self.lambd * F_S_tilde
        return total_loss


class FalsePositiveMin(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, lamma_tv=1, lamma_df=1, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lamma_tv = lamma_tv
        self.lamma_dicefocal = lamma_df
        self.smooth = smooth
        self.dicefocal = DiceFocalLoss(to_onehot_y=True, softmax=True)
        
    def forward(self, predictions, labels):
        preds = F.softmax(predictions, dim=1)
        target = one_hot_encoding(labels)        
        tversky_loss = tverskyLoss(preds, target, alpha=self.alpha, beta=self.beta, smooth=self.smooth)
        dfl_loss = self.dicefocal(predictions, labels)
        total_loss = self.lamma_dicefocal*dfl_loss + self.lamma_tv*tversky_loss
        return total_loss

class RCELoss(nn.Module):
    def __init__(self, lambda_param=1.0):
        """
        Initialize the RCE loss function.

        Parameters:
        lambda_param (float): Balancing weight for the L1 regularization term.
        """
        super(RCELoss, self).__init__()
        self.lambda_param = lambda_param
        self.ce_loss = FocalLoss()

    def forward(self, outputs, targets):
        """
        Compute the RCE loss.

        Parameters:
        outputs (torch.Tensor): Predictions from the network (logits).
        targets (torch.Tensor): Ground truth labels.

        Returns:
        torch.Tensor: Computed RCE loss.
        """
        one_hot_targets= one_hot_encoding(targets)
        outputs = torch.softmax(outputs, dim=1)
        # Compute cross-entropy loss

        ce_loss = self.ce_loss(outputs, one_hot_targets, gamma=2)

        # Convert logits to probabilities
        num_classes = outputs.size(1)

        # Compute L1 norm regularization term with class weighting
        l1_reg = torch.abs(one_hot_targets - outputs)
        targetss = targets.view(-1).long()

        # Calculate class weights based on inverse class frequency
        class_counts = torch.bincount(targetss.long(), minlength=num_classes).float()
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum()
        voxel_weights = class_weights[targets.long()]
        voxel_weights = voxel_weights.expand(-1, num_classes, -1, -1, -1)

        # Apply class weights to the L1 regularization term
        weighted_l1_reg = (voxel_weights * l1_reg).sum(dim=1).mean()

        # Combine cross-entropy and weighted L1 regularization term
        rce_loss = ce_loss + self.lambda_param * weighted_l1_reg

        return rce_loss
    

def prf_loss(gamma=-2.0, epsilon=1e-6):
    """
    Precision-Recall Focal Loss function for binary segmentation
    
    Parameters
    ----------
    gamma : float, optional
        focal parameter controls the degree of focusing on hard samples, by default 2.0
    epsilon : float, optional
        smoothing constant to prevent log(0), by default 1e-6
    """
    def loss_function(y_preds, y_true):
        # Apply softmax to y_pred to get probabilities
        y_pred = F.softmax(y_preds, dim=1)
        
        # One-hot encode y_true
        y_true_one_hot = one_hot_encoding(y_true)
        
        axis = (1, 2, 3)
        # Calculate true positives (tp), false negatives (fn) and false positives (fp) for the positive class
        tp = torch.sum(y_true_one_hot[:, 1] * y_pred[:, 1], dim=axis)
        fn = torch.sum(y_true_one_hot[:, 1] * (1 - y_pred[:, 1]), dim=axis)
        fp = torch.sum((1 - y_true_one_hot[:, 1]) * y_pred[:, 1], dim=axis)
        
        # Calculate Precision and Recall
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        # Calculate PRF Loss
        prf_loss = - ((precision+epsilon) ** gamma) * torch.log(recall + epsilon)
        
        # Average class scores
        prf_loss = torch.mean(prf_loss)

        return prf_loss
    
    return loss_function


class GHMLoss(nn.Module):
    def __init__(self, bins=10, alpha=0.75):
        super(GHMLoss, self).__init__()
        self.bins = bins
        self.alpha = alpha
        self.edges = torch.linspace(0, 1, bins + 1)

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        gradients = torch.abs(probabilities.detach() - targets)
        weights = torch.zeros_like(logits)
        
        for i in range(self.bins):
            bin_mask = (gradients >= self.edges[i]) & (gradients < self.edges[i+1])
            bin_count = bin_mask.sum().item() + 1e-6  # to avoid division by zero
            weights[bin_mask] = self.alpha / bin_count

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weighted_bce_loss = bce_loss * weights

        return weighted_bce_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=[2, 3, 4])
        dice = (2. * intersection + self.smooth) / (
            probabilities.sum(dim=[2, 3, 4]) + targets.sum(dim=[2, 3, 4]) + self.smooth)

        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(CombinedLoss, self).__init__()
        self.ghm_loss = GHMLoss()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, targets):
        targets_one_hot = F.one_hot(targets.long().squeeze(1), num_classes=2)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

        loss_ghm = self.ghm_loss(logits, targets_one_hot)
        loss_focal = self.focal_loss(logits, targets_one_hot)
        loss_dice = self.dice_loss(logits, targets_one_hot)

        return self.alpha * loss_ghm + self.beta * loss_focal + self.gamma * loss_dice

class GHMDiceLoss(nn.Module):
    def __init__(self, gamma=0.0, bins=10, epsilon=0.1, g_lambda=1.0, f_lambda=1.0):
        super(GHMDiceLoss, self).__init__()
        self.gamma = gamma
        self.bins = bins
        self.epsilon = epsilon
        self.edges = torch.linspace(0, 1, bins + 1).cuda()
        self.g_lambda = g_lambda
        self.f_lambda = f_lambda
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
        counts = []
        for center in self.edges:
            mask = (g >= center - self.epsilon/2) & (g < center + self.epsilon/2)
            count_in_region = mask.sum().item()
            valid_length = min(center + self.epsilon/2, 1) - max(center - self.epsilon/2, 0)
            gd[mask] = (count_in_region / valid_length).type(torch.float64)
            counts.append(count_in_region / valid_length)
        beta = N / (gd + 1e-8)
        beta = beta.view_as(gradients)
        weighted_intersection = (beta * probabilities * targets).sum(dim=[2, 3, 4])
        weighted_union = (beta*(probabilities**2 + targets**2)).sum(dim=[2, 3, 4])
        dice_score = (2. * weighted_intersection + 1e-8) / (weighted_union + 1e-8)
        loss = (1 - dice_score.mean())+focal_loss
        return loss
    
class GHMDiceLoss1(nn.Module):
    def __init__(self, gamma=0.0, bins=10, epsilon=0.1, g_lambda=1.0, f_lambda=1.0):
        super(GHMDiceLoss1, self).__init__()
        device = torch.device('cuda:1')
        self.gamma = gamma
        self.bins = bins
        self.epsilon = epsilon
        self.edges = torch.linspace(0, 1, bins + 1).to(device)
        self.g_lambda = g_lambda
        self.f_lambda = f_lambda
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
        counts = []
        for center in self.edges:
            mask = (g >= center - self.epsilon/2) & (g < center + self.epsilon/2)
            count_in_region = mask.sum().item()
            valid_length = min(center + self.epsilon/2, 1) - max(center - self.epsilon/2, 0)
            gd[mask] = (count_in_region / valid_length).type(torch.float64)
            counts.append(count_in_region / valid_length)
        beta = N / (gd + 1e-8)
        beta = beta.view_as(gradients)
        weighted_intersection = (beta * probabilities * targets).sum(dim=[2, 3, 4])
        weighted_union = (beta*(probabilities**2 + targets**2)).sum(dim=[2, 3, 4])
        dice_score = (2. * weighted_intersection + 1e-8) / (weighted_union + 1e-8)
        loss = (1 - dice_score.mean())+focal_loss
        return loss
    
class fnrghdl(nn.Module):
    def __init__(self, lamma=1, c=15):
        super(fnrghdl, self).__init__()
        self.lamma = lamma
        self.ghdlfl = GHMDiceLoss()
        self.c = torch.tensor(c)
        
    def forward(self, logits, labels, epoch):
        ghdlfl = self.ghdlfl(logits, labels)
        probs = torch.softmax(logits, dim=1)
        targets = F.one_hot(labels.long().squeeze(1), num_classes= 2).permute(0,4,1,2,3).float()
        
        S_tilde = (1 / (1 + torch.exp(-(probs - 0.5) * self.c)) - 0.5) * \
            (1 + torch.exp(-0.5 * self.c)) / (1 - torch.exp(-0.5 * self.c)) + 0.5
        F_S_tilde_list = []
        for class_index in range(probs.shape[1]):
            sum_y = targets[:, class_index].sum()
            intersection = (S_tilde[:, class_index] > 0.5).float() * (targets[:, class_index]).float()
            sum_intersection = intersection.sum()
            F_S_tilde_class = torch.abs(sum_intersection - sum_y) / (sum_y+1e-6)
            F_S_tilde_list.append(F_S_tilde_class)
        F_S_tilde = torch.stack(F_S_tilde_list).mean()
        if epoch > 300:
            loss =ghdlfl+F_S_tilde
        else:
            loss =ghdlfl
        return loss        
        
        
class GHMDiceLossNorm(nn.Module):
    def __init__(self, gamma=2.0, bins=10, epsilon=0.1, update_interval=100, alpha=0.9, lambda_smoothing=0.75):
        super(GHMDiceLossNorm, self).__init__()
        self.gamma = gamma
        self.bins = bins
        self.epsilon = epsilon
        self.update_interval = update_interval
        self.alpha = alpha
        self.lambda_smoothing = lambda_smoothing
        # self.edges = None
        self.edges = torch.linspace(0, 1, bins + 1).cuda()
        # self.step = 0
        self.smoothed_gd = None

    # def update_bins(self, gradients):
    #     # Compute new edges based on current gradients
    #     new_edges = torch.quantile(gradients, torch.linspace(0, 1, self.bins + 1).cuda()).detach()
    #     if self.edges is None:
    #         self.edges = new_edges
    #     else:
    #         # Smooth update using a moving average
    #         self.edges = self.alpha * self.edges + (1 - self.alpha) * new_edges

    def update_gradient_density(self, gd_current):
        if self.smoothed_gd is None:
            self.smoothed_gd = gd_current.clone()
        elif self.smoothed_gd.size() == gd_current.size():
            self.smoothed_gd = self.lambda_smoothing * self.smoothed_gd + (1 - self.lambda_smoothing) * gd_current

    def forward(self, logits, labels):
        probabilities = torch.sigmoid(logits)
        targets = one_hot_encoding(labels)
        gradients = torch.abs(probabilities - targets)
        g = gradients.view(-1)

        # # Update the bin edges periodically
        # if self.step % self.update_interval == 0:
        #     self.update_bins(gradients)
        # self.step += 1

        N = g.size(0)
        gd_current = torch.zeros_like(g)
        for center in self.edges:
            mask = (g >= center - self.epsilon/2) & (g < center + self.epsilon/2)
            count_in_region = mask.sum().item()
            valid_length = min(center + self.epsilon/2, 1) - max(center - self.epsilon/2, 0)
            gd_current[mask] = count_in_region / valid_length
        self.update_gradient_density(gd_current)
        
        if self.smoothed_gd.size() != gd_current.size():
            beta = N / (gd_current + 1e-6)
        else:
            beta = N / (self.smoothed_gd + 1e-6)
        beta = beta.view_as(gradients)

        # Normalize beta across each batch
        # beta_sum = beta.sum(dim=[2, 3, 4], keepdim=True)
        # beta_norm = beta / (beta_sum + 1e-6)  # Adding a small constant to avoid division by zero

        weighted_intersection = (beta * probabilities * targets).sum(dim=[2, 3, 4])
        weighted_union = (beta * (probabilities + targets)).sum(dim=[2, 3, 4])

        dice_score = (2. * weighted_intersection + 1e-6) / (weighted_union + 1e-6)
        return 1 - dice_score.mean()


class GHDLoss(nn.Module):
    def __init__(self, epsilon=0.2):
        super(GHDLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        probs = F.torch.softmax(logits,1,torch.float64)
        if targets.shape[1] == 1:
            labels = one_hot_encoding(targets).type(torch.float64)
        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)
        labels_flat = labels.view(labels.shape[0], labels.shape[1], -1)

        # Calculate gradient norms
        s = torch.sum(probs_flat, dim=2) + torch.sum(labels_flat, dim=2)
        intersection = torch.sum(probs_flat * labels_flat, dim=2)
        iou_ratio = (2 * (intersection+1e-6) / (s+1e-6)).unsqueeze(2)  # Fix: Ensure correct dimension for broadcasting
        # Calculate the absolute gradient norm difference
        g_star = torch.abs(labels_flat - iou_ratio * probs_flat)
        N = g_star.flatten().size(0)

        # Calculate gradient density
        bins = (1 / self.epsilon) * (1 + torch.max(g_star))
        hist = torch.histc(g_star, bins=int(bins), min=0, max=1)
        density = hist /self.epsilon

        # Weights calculation
        weights =  N/ (density[torch.floor(g_star / self.epsilon).long()]+1e-6)

        # Calculate the weighted Dice coefficient
        weighted_intersection = torch.sum(weights * probs_flat * labels_flat, dim=2)
        weighted_union = torch.sum(weights * (probs_flat**2 + labels_flat**2), dim=2)

        # Dice score calculation
        dice_score = 2 * (weighted_intersection+1e-6)/ (weighted_union +1e-6)

        # GHDL loss is 1 minus the mean Dice score
        return 1 - dice_score.mean()