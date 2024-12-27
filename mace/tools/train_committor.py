###########################################################################################
# Training script
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage
from torchmetrics import Metric

from . import torch_geometric
from .checkpoint import CheckpointHandler, CheckpointState
from .torch_tools import TensorDict, TensorDictList, to_numpy
from .utils import MetricsLogger


@dataclasses.dataclass
class SWAContainer:
    model: AveragedModel
    scheduler: SWALR
    start: int
    loss_fn: torch.nn.Module


def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    epoch=None,
):
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    logger.log(eval_metrics)
    if epoch is None:
        inintial_phrase = "Initial"
    else:
        inintial_phrase = f"Epoch {epoch}"
    logging.info(
        f"{inintial_phrase}: loss={valid_loss:8.4f}",
    )


def train_committor(
    model: torch.nn.Module,
    loss_fn_train: torch.nn.Module,
    loss_fn_valid: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    checkpoint_handler_2: CheckpointHandler,
    logger: MetricsLogger,
    eval_interval: int,
    output_args: Dict[str, bool],
    device: torch.device,
    save_interval: int = 10,
    swa: Optional[SWAContainer] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    max_grad_norm: Optional[float] = 10.0,
    log_wandb: bool = False,
    wall_clock_time: float = 0,
    distributed: bool = False,
    distributed_model: Optional[DistributedDataParallel] = None,
    train_sampler: Optional[DistributedSampler] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    readouts_only: bool = False,
):
    # Start timers if wanted
    if wall_clock_time != 0:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    lowest_loss = np.inf
    valid_loss = np.inf
    patience_counter = 0
    swa_start = True
    keep_last = False
    if log_wandb:
        import wandb

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")
    logging.info("Started training")
    # See if constant shift in model's sigmoid needs to be adjusted
    if model.psigmoid.c == 0.0:
        logging.info("Adjusting constant shift in model's sigmoid")
        new_shift = adjust_sigmoid_shift(
            model=model,
            data_loader=train_loader,
            output_args=output_args,
            device=device,
            distributed_model=distributed_model,
            rank=rank,
            distributed=distributed,
        )
        model.psigmoid.update_c(new_shift * model.psigmoid.p)
        logging.info(f"New constant shift in model's sigmoid is {model.psigmoid.c}")
    logging.info("Loss metrics on validation set")
    epoch = start_epoch
    valid_loss = 0.0
    valid_loss, eval_metrics = evaluate(
        model=model,
        loss_fn=loss_fn_valid,
        data_loader=valid_loader,
        output_args=output_args,
        device=device,
        readouts_only=readouts_only,
    )
    logging.info(f"Initial loss: {valid_loss}")
    logging.info(eval_metrics)
    if start_epoch == 0:
        if (distributed and rank == 0) or not distributed:
            valid_err_log(
                valid_loss,
                eval_metrics,
                logger,
                None,
            )
    else:
        if (distributed and rank == 0) or not distributed:
            valid_err_log(
                valid_loss,
                eval_metrics,
                logger,
                start_epoch,
            )

    while epoch < max_num_epochs:
        # Check time
        if wall_clock_time != 0:
            end.record()
            torch.cuda.synchronize()
            if start.elapsed_time(end) / 1000 > wall_clock_time:
                logging.info(
                    f"Stopping optimization after {wall_clock_time} seconds of wall_clock time"
                )
                # Save the model
                if ema is not None:
                    with ema.average_parameters():
                        keep_last = False
                        checkpoint_handler.save(
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                            keep_last=keep_last,
                        )
                else:
                    keep_last = False
                    checkpoint_handler.save(
                        state=CheckpointState(model, optimizer, lr_scheduler),
                        epochs=epoch,
                        keep_last=keep_last,
                    )
                break
        # LR scheduler and SWA update
        if swa is None or epoch < swa.start:
            if epoch > start_epoch:
                lr_scheduler.step(
                    metrics=valid_loss
                )  # Can break if exponential LR, TODO fix that!
        else:
            if swa_start:
                logging.info("Changing loss based on SWA")
                lowest_loss = np.inf
                swa_start = False
                keep_last = True
            loss_fn_train = swa.loss_fn
            swa.model.update_parameters(model)
            if epoch > start_epoch:
                swa.scheduler.step()

        # Train
        if distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            model=model,
            output_args=output_args,
            loss_fn=loss_fn_train,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            max_grad_norm=max_grad_norm,
            ema=ema,
            logger=logger,
            device=device,
            distributed_model=distributed_model,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
        )
        if distributed:
            torch.distributed.barrier()

        # Validate
        if epoch % eval_interval == 0:
            model_to_evaluate = (
                model if distributed_model is None else distributed_model
            )
            param_context = (
                ema.average_parameters() if ema is not None else nullcontext()
            )
            with param_context:
                valid_loss = 0.0
                wandb_log_dict = {}
                valid_loss, eval_metrics = evaluate(
                    model=model_to_evaluate,
                    loss_fn=loss_fn_valid,
                    data_loader=valid_loader,
                    output_args=output_args,
                    device=device,
                    readouts_only=readouts_only,
                )
                if (distributed and rank == 0) or not distributed:
                    valid_err_log(
                        valid_loss,
                        eval_metrics,
                        logger,
                        epoch,
                    )
                    if log_wandb:
                        wandb_log_dict = {
                            "epoch": epoch,
                            "valid_loss": valid_loss,
                        }
                        wandb.log(wandb_log_dict)

            if (distributed and rank == 0) or not distributed:
                if valid_loss >= lowest_loss:
                    patience_counter += 1
                    if swa is not None:
                        if patience_counter >= patience and epoch < swa.start:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement and starting swa"
                            )
                            epoch = swa.start
                    elif patience_counter >= patience:
                        logging.info(
                            f"Stopping optimization after {patience_counter} epochs without improvement"
                        )
                        break
                else:
                    lowest_loss = valid_loss
                    patience_counter = 0
                    if ema is not None:
                        with ema.average_parameters():
                            checkpoint_handler.save(
                                state=CheckpointState(model, optimizer, lr_scheduler),
                                epochs=epoch,
                                keep_last=keep_last,
                            )
                            keep_last = False
                    else:
                        checkpoint_handler.save(
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                            keep_last=keep_last,
                        )
                        keep_last = False
                if epoch % save_interval == 0:
                    if ema is not None:
                        with ema.average_parameters():
                            checkpoint_handler_2.save(
                                state=CheckpointState(model, optimizer, lr_scheduler),
                                epochs=epoch,
                                keep_last=True,
                            )
                    else:
                        checkpoint_handler_2.save(
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                            keep_last=True,
                        )

        if distributed:
            torch.distributed.barrier()
            # Communicate epoch to all processes
            torch.distributed.broadcast(
                torch.tensor([epoch], device=device, dtype=torch.int), src=0
            )
        epoch += 1

    logging.info("Training complete")


def adjust_sigmoid_shift(
    model: torch.nn.Module,
    data_loader: DataLoader,
    output_args: Dict[str, bool],
    device: torch.device,
    distributed_model: Optional[DistributedDataParallel] = None,
    rank: Optional[int] = 0,
    distributed: bool = False,
) -> float:
    model_to_train = model if distributed_model is None else distributed_model
    total_contribution = torch.zeros(1, device=device, dtype=torch.get_default_dtype())
    num_samples = torch.zeros(1, device=device, dtype=torch.int)
    for batch in data_loader:
        if output_args["training_loss"] == "training":
            atomic_data, _, _ = batch
        elif output_args["training_loss"] == "validation":
            atomic_data, _ = batch
        atomic_data = atomic_data.to(device)
        atomic_data_dict = atomic_data.to_dict()
        output = model_to_train(atomic_data_dict)
        total_contribution += torch.sum(output["total_contributions"]).detach()
        num_samples += output["total_contributions"].shape[0]
        break
    if distributed:
        torch.distributed.all_reduce(total_contribution)
        torch.distributed.all_reduce(num_samples)
    total_contribution /= num_samples
    if (distributed and rank == 0) or not distributed:
        logging.info(
            f"Adjusting constant shift in model's sigmoid to {total_contribution.item()}"
        )
    return total_contribution.item()


def train_one_epoch(
    model: torch.nn.Module,
    output_args: Dict[str, bool],
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    max_grad_norm: Optional[float],
    ema: Optional[ExponentialMovingAverage],
    logger: MetricsLogger,
    device: torch.device,
    distributed_model: Optional[DistributedDataParallel] = None,
    rank: Optional[int] = 0,
    world_size: Optional[int] = 1,
    distributed: bool = False,
) -> None:
    model_to_train = model if distributed_model is None else distributed_model
    for batch in data_loader:
        _, opt_metrics = take_step(
            model=model_to_train,
            output_args=output_args,
            loss_fn=loss_fn,
            batch=batch,
            optimizer=optimizer,
            ema=ema,
            max_grad_norm=max_grad_norm,
            device=device,
            world_size=world_size,
            distributed=distributed,
        )
        opt_metrics["mode"] = "opt"
        opt_metrics["epoch"] = epoch
        if (distributed and rank == 0) or not distributed:
            logger.log(opt_metrics)


def take_step(
    model: torch.nn.Module,
    output_args: Dict[str, bool],
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    max_grad_norm: Optional[float],
    device: torch.device,
    world_size: int = 1,
    distributed: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    loss = 0.0
    if output_args["training_loss"] == "training":
        atomic_data, cv_data, atomic_sub_data = batch
        atomic_data = atomic_data.to(device)
        cv_data = cv_data.to(device)
        optimizer.zero_grad(set_to_none=True)
        atomic_data_dict = atomic_data.to_dict()
        output = model(
            atomic_data_dict,
        )
        output_sub = []
        for atomic_sub in atomic_sub_data:
            atomic_sub = atomic_sub.to(device)
            atomic_sub_dict = atomic_sub.to_dict()
            output_sub_ = model(
                atomic_sub_dict,
            )
            for key, value in output_sub_.items():
                if isinstance(value, torch.Tensor):
                    output_sub_.update({key: value.detach()})
            output_sub.append(output_sub_)
        logging.info(f"Committor output: {output['committor']}")
        loss = loss_fn(output=output, output_sub=output_sub, cv_dt=cv_data)
    elif output_args["training_loss"] == "validation":
        atomic_data, committor = batch
        atomic_data = atomic_data.to(device)
        committor = committor.to(device)
        optimizer.zero_grad(set_to_none=True)
        atomic_data_dict = atomic_data.to_dict()
        output = model(
            atomic_data_dict,
        )
        logging.info(f"Committor output: {output['committor']}")
        logging.info(f"Committor ref: {committor}")
        loss = loss_fn(output=output, committor_ref=committor)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    if ema is not None:
        ema.update()

    # get loss across all processes
    loss = loss.detach()
    if distributed:
        torch.distributed.all_reduce(loss)
        loss /= world_size

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    output_args: Dict[str, bool],
    device: torch.device,
    readouts_only: bool = False,
) -> Tuple[float, Dict[str, Any]]:

    if readouts_only:
        if not isinstance(model, DistributedDataParallel):
            model.disable_grad_readout()
        else:
            model.module.disable_grad_readout()
    else:
        for param in model.parameters():
            param.requires_grad = False

    metrics = MACELoss(loss_fn=loss_fn, output_args=output_args).to(device)

    start_time = time.time()
    if output_args["validation_loss"] == "training":
        for batch in data_loader:
            atomic_data, cv_data, atomic_sub_data = batch
            atomic_data = atomic_data.to(device)
            cv_data = cv_data.to(device)
            atomic_data_dict = atomic_data.to_dict()
            output = model(
                atomic_data_dict,
            )
            output_sub = []
            for atomic_sub in atomic_sub_data:
                atomic_sub_dict = atomic_sub.to_dict()
                output_sub_ = model(
                    atomic_sub_dict,
                )
                for key, value in output_sub_.items():
                    if isinstance(value, torch.Tensor):
                        output_sub_.update({key: value.detach()})
                output_sub.append(output_sub_)
            avg_loss, aux = metrics(
                output=output, output_sub=output_sub, cv_data=cv_data
            )
    elif output_args["validation_loss"] == "validation":
        for batch in data_loader:
            atomic_data, committor = batch
            atomic_data = atomic_data.to(device)
            committor = committor.to(device)
            atomic_data_dict = atomic_data.to_dict()
            output = model(
                atomic_data_dict,
            )
            avg_loss, aux = metrics(output=output, committor=committor)

    avg_loss, aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    if readouts_only:
        if not isinstance(model, DistributedDataParallel):
            model.enable_grad_readout()
        else:
            model.module.enable_grad_readout()
    else:
        for param in model.parameters():
            param.requires_grad = True

    return avg_loss, aux


class MACELoss(Metric):
    def __init__(self, loss_fn: torch.nn.Module, output_args: Dict[str, bool]):
        super().__init__()
        self.loss_fn = loss_fn
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_data", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.loss_type = "training"
        if output_args["validation_loss"] == "validation":
            self.loss_type = "validation"

    def update(
        self,
        output: TensorDict,
        output_sub: TensorDictList = None,
        cv_data: torch.Tensor = None,
        committor: torch.Tensor = None,
    ):  # pylint: disable=arguments-differ
        loss = 0.0
        if self.loss_type == "training":
            loss = self.loss_fn(output=output, output_sub=output_sub, cv_data=cv_data)
        elif self.loss_type == "validation":
            loss = self.loss_fn(output=output, committor_ref=committor)
        self.total_loss += loss
        self.num_data += 1

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return to_numpy(delta)

    def compute(self):
        aux = {}
        aux["loss"] = to_numpy(self.total_loss / self.num_data).item()
        return aux["loss"], aux
