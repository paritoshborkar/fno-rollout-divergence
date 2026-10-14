from abc import ABC, abstractmethod

import torch


class WassersteinP1DLoss(ABC):
    """
    Differentiable Wasserstein-p 1D distance between predicted and target field
    values, treating each sample's flattened (C, H, W) values as an empirical 1D
    distribution.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-5):
        assert reduction in ("sum", "mean")
        self.reduction = reduction
        self.eps = eps

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError()


class WassersteinOne1DLoss(WassersteinP1DLoss):
    """
    Differentiable Wasserstein-1 1D distance
    For equal-size, uniformly-weighted distributions this reduces to 
    the mean absolute difference between their sorted values
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-5):
        super().__init__(reduction=reduction, eps=eps)

    @property
    def name(self):
        return "1-wasserstein_1d"

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        pred_sorted, _ = torch.sort(y_pred.flatten(start_dim=1), dim=-1)
        target_sorted, _ = torch.sort(y.flatten(start_dim=1), dim=-1)

        per_sample = (torch.abs(pred_sorted - target_sorted)).mean(dim=-1)  # (N,)
        return per_sample.sum() if self.reduction == "sum" else per_sample.mean()


class WassersteinTwo1DLoss(WassersteinP1DLoss):
    """
    Differentiable Wasserstein-2 1D distance 
    For equal-size, uniformly-weighted distributions this reduces to the 
    RMSE difference between their sorted values
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-5):
            super().__init__(reduction=reduction, eps=eps)

    @property
    def name(self):
        return "2-wasserstein_1d"

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        pred_sorted, _ = torch.sort(y_pred.flatten(start_dim=1), dim=-1)
        target_sorted, _ = torch.sort(y.flatten(start_dim=1), dim=-1)

        per_sample = torch.sqrt(
            ((pred_sorted - target_sorted) ** 2).mean(dim=-1) + self.eps
        )  # (N,)
        return per_sample.sum() if self.reduction == "sum" else per_sample.mean()
