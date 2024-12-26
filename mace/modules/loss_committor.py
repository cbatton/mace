###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Dict, List, Literal, Tuple, Union

import torch

from mace.tools import TensorDict, TensorDictList

ConditionType = Union[
    Tuple[Literal["lt", "gt"], float],
    Tuple[Literal["between"], float, float],
    Tuple[Literal["periodic"], float, float],
]
DimensionCondition = Tuple[int, ConditionType]
ConditionsDict = Dict[Literal["0", "1"], List[DimensionCondition]]


def committor_loss_train(
    committor: torch.Tensor, committor_dt: torch.Tensor
) -> torch.Tensor:
    return torch.mean((committor - committor_dt) ** 2)


def committor_loss_train_log(
    committor: torch.Tensor, committor_dt: torch.Tensor
) -> torch.Tensor:
    return torch.mean(
        0.5 * (torch.log(committor) - torch.log(committor_dt)) ** 2
        + 0.5 * (torch.log(1 - committor) - torch.log(1 - committor_dt)) ** 2
    )


def committor_loss_valid(
    committor_pred: torch.Tensor, committor_ref: torch.Tensor
) -> torch.Tensor:
    return torch.mean((committor_pred - committor_ref) ** 2)


class CommittorTrainingLoss(torch.nn.Module):
    def __init__(self, train_type="log", conditions: ConditionsDict = None) -> None:
        super().__init__()
        self.train_type = train_type
        if train_type == "log":
            self.loss = committor_loss_train_log
        elif train_type == "mse":
            self.loss = committor_loss_train
        else:
            raise ValueError(f"Unknown train type {train_type}")
        self.conditions = conditions

    def forward(
        self, output: TensorDict, output_sub: TensorDictList, cv_dt: torch.Tensor
    ) -> torch.Tensor:
        # Extract out committor and committor_dt
        committor = output["committor"]
        committor_dt = torch.stack([o["committor"] for o in output_sub], dim=-1)
        # apply conditions, then take mean of committor_dt along non-batch dimensions
        if self.conditions is not None:
            committor_dt = apply_conditional_thresholds(
                cv_dt, committor_dt, self.conditions
            )
        committor_dt = committor_dt.mean(dim=-1)
        return self.loss(committor, committor_dt)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class CommittorValidationLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, output: TensorDict, committor_ref: torch.Tensor) -> torch.Tensor:
        # detach, take mean of committor_dt along non-batch dimensions
        committor_pred = output["committor"]
        return committor_loss_valid(committor_pred, committor_ref)

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
            elif condition[0] == "periodic":
                dim_mask = (x[..., dim] < condition[1]) | (x[..., dim] > condition[2])
            else:
                raise ValueError(f"Unknown condition type: {condition[0]}")

            if value == "0":
                mask_0 &= dim_mask
            elif value == "1":
                mask_1 &= dim_mask

    result = torch.where(mask_0, torch.zeros_like(y), y)
    result = torch.where(mask_1, torch.ones_like(y), result)

    return result
