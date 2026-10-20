import torch


class CompositeLoss:
    """
    Composition of multiple losses, weighted and summed
    """

    def __init__(self, loss_dist: list[tuple[float, object]]):
        self.loss_dist = loss_dist


    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculate weighted loss for each loss function
        """

        total_loss = 0
        for weight, loss in self.loss_dist:
            total_loss += weight * loss(y_pred=y_pred, y=y, **kwargs)

        return total_loss