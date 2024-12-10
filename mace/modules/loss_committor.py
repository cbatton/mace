###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Dict, List, Literal, Tuple, Union

import torch

ConditionType = Union[
    Tuple[Literal["lt", "gt"], float], Tuple[Literal["between"], float, float]
]
DimensionCondition = Tuple[int, ConditionType]
ConditionsDict = Dict[Literal["0", "1"], List[DimensionCondition]]


def committor_loss_train(cv: torch.Tensor, cv_dt: torch.Tensor) -> torch.Tensor:
    return torch.mean((cv - cv_dt) ** 2)


def committor_loss_train_log(cv: torch.Tensor, cv_dt: torch.Tensor) -> torch.Tensor:
    return torch.mean(
        0.5 * (torch.log(cv) - torch.log(cv_dt)) ** 2
        + 0.5 * (torch.log(1 - cv) - torch.log(1 - cv_dt)) ** 2
    )


class CommittorTrainingLoss(torch.nn.Module):
    def __init__(self, train_type="log") -> None:
        super().__init__()
        self.train_type = train_type
        if train_type == "log":
            self.loss = committor_loss_train_log
        elif train_type == "mse":
            self.loss = committor_loss_train
        else:
            raise ValueError(f"Unknown train type {train_type}")

    def forward(self, cv: torch.Tensor, cv_dt: torch.Tensor) -> torch.Tensor:
        # detach, take mean of cv_dt along non-batch dimensions
        cv_dt = cv_dt.mean(dim=-1)
        return self.loss(cv, cv_dt)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def apply_conditional_thresholds(
    x: torch.Tensor, y: torch.Tensor, conditions: ConditionsDict
) -> torch.Tensor:
    """
    Apply conditions to x and modify y accordingly.

    Args:
    x (torch.Tensor): Input tensor of shape [..., D] where D is the number of dimensions for thresholds
    y (torch.Tensor): Tensor to be modified, should be broadcastable with x[..., 0]
    conditions (ConditionsDict): Dictionary with keys '0' and '1', each containing a list of conditions

    Returns:
    torch.Tensor: Modified y tensor
    """
    mask_0 = torch.ones_like(y, dtype=torch.bool)
    mask_1 = torch.ones_like(y, dtype=torch.bool)

    # Ensure x one dimension higher than y
    if x.dim() == y.dim():
        x = x.unsqueeze(-1)

    for value, condition_list in conditions.items():
        for dim, condition in condition_list:
            if condition[0] == "lt":
                dim_mask = x[..., dim] < condition[1]
            elif condition[0] == "gt":
                dim_mask = x[..., dim] > condition[1]
            elif condition[0] == "between":
                dim_mask = (x[..., dim] > condition[1]) & (x[..., dim] < condition[2])
            else:
                raise ValueError(f"Unknown condition type: {condition[0]}")

            if value == "0":
                mask_0 &= dim_mask
            elif value == "1":
                mask_1 &= dim_mask

    result = torch.where(mask_0, torch.zeros_like(y), y)
    result = torch.where(mask_1, torch.ones_like(y), result)

    return result
