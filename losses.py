import numpy as np
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal loss for binary (2-class) classification."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a = torch.zeros(batch_size, 2, device=x.device).scatter_(1, y.unsqueeze(1), 1)
    y_b = torch.zeros(batch_size, 2, device=x.device).scatter_(1, y[index].unsqueeze(1), 1)
    soft_labels = lam * y_a + (1 - lam) * y_b
    return mixed_x, soft_labels


def build_hybrid_criterion(class_weights, label_smoothing, gamma, focal_weight):
    """Hybrid criterion: (1 - focal_weight) * CE  +  focal_weight * Focal."""
    ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    focal = FocalLoss(gamma=gamma, weight=class_weights, label_smoothing=label_smoothing)

    def hybrid(logits, labels):
        return (1 - focal_weight) * ce(logits, labels) + focal_weight * focal(logits, labels)

    return hybrid
