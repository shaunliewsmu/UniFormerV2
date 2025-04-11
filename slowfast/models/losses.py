#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from slowfast.models.focal_loss import FocalLoss


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """
    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "focal_loss": FocalLoss,
}


def get_loss_func(loss_name, cfg=None):
    """
    Retrieve the loss given the loss name.
    Args:
        loss_name (str): name of the loss to retrieve.
        cfg (CfgNode): optional config node containing loss parameters.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    
    # For regular losses, just return the loss class
    if loss_name != "focal_loss":
        return _LOSSES[loss_name]
    
    # Special handling for focal loss to allow passing alpha and gamma parameters
    def focal_loss_wrapper(**kwargs):
        # Default values
        alpha = 0.25
        gamma = 2.0
        
        # Override with config values if provided
        if cfg is not None and hasattr(cfg.MODEL, 'FOCAL_LOSS'):
            alpha = cfg.MODEL.FOCAL_LOSS.ALPHA
            gamma = cfg.MODEL.FOCAL_LOSS.GAMMA
        
        # Create focal loss with parameters
        return FocalLoss(alpha=alpha, gamma=gamma, **kwargs)
    
    return focal_loss_wrapper
