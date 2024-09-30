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
from .torch_tools import to_numpy
from .utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
)


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
    log_errors,
    forces=False,
    epoch=None,
):
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    logger.log(eval_metrics)
    if epoch is None:
        inintial_phrase = "Initial"
    else:
        inintial_phrase = f"Epoch {epoch}"
    if log_errors == "PerAtomRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        if forces:
            error_f = eval_metrics["rmse_f"] * 1e3
            logging.info(
                f"{inintial_phrase}: loss={valid_loss:8.4f}, RMSE_E_per_atom={error_e:8.3f} meV, RMSE_F={error_f:8.3f} meV / A"
            )
        else:
            logging.info(
                f"{inintial_phrase}: loss={valid_loss:8.4f}, RMSE_E_per_atom={error_e:8.3f} meV"
            )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_stress"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_stress = eval_metrics["rmse_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.4f}, RMSE_E_per_atom={error_e:8.3f} meV, RMSE_F={error_f:8.3f} meV / A, RMSE_stress={error_stress:8.3f} meV / A^3",
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_virials = eval_metrics["rmse_virials_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.4f}, RMSE_E_per_atom={error_e:8.3f} meV, RMSE_F={error_f:8.3f} meV / A, RMSE_virials_per_atom={error_virials:8.3f} meV",
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_stress_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_stress = eval_metrics["mae_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.4f}, MAE_E_per_atom={error_e:8.3f} meV, MAE_F={error_f:8.3f} meV / A, MAE_stress={error_stress:8.3f} meV / A^3"
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_virials = eval_metrics["mae_virials"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.4f}, MAE_E_per_atom={error_e:8.3f} meV, MAE_F={error_f:8.3f} meV / A, MAE_virials={error_virials:8.3f} meV"
        )
    elif log_errors == "TotalRMSE":
        error_e = eval_metrics["rmse_e"] * 1e3
        if forces:
            error_f = eval_metrics["rmse_f"] * 1e3
            logging.info(
                f"{inintial_phrase}: loss={valid_loss:8.4f}, RMSE_E={error_e:8.3f} meV, RMSE_F={error_f:8.3f} meV / A"
            )
        else:
            logging.info(
                f"{inintial_phrase}: loss={valid_loss:8.4f}, RMSE_E={error_e:8.3f} meV"
            )
    elif log_errors == "PerAtomMAE":
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        if forces:
            error_f = eval_metrics["mae_f"] * 1e3
            logging.info(
                f"{inintial_phrase}: loss={valid_loss:8.4f}, MAE_E_per_atom={error_e:8.3f} meV, MAE_F={error_f:8.3f} meV / A"
            )
        else:
            logging.info(
                f"{inintial_phrase}: loss={valid_loss:8.4f}, MAE_E_per_atom={error_e:8.3f} meV"
            )
    elif log_errors == "TotalMAE":
        error_e = eval_metrics["mae_e"] * 1e3
        if forces:
            error_f = eval_metrics["mae_f"] * 1e3
            logging.info(
                f"{inintial_phrase}: loss={valid_loss:8.4f}, MAE_E={error_e:8.3f} meV, MAE_F={error_f:8.3f} meV / A"
            )
        else:
            logging.info(
                f"{inintial_phrase}: loss={valid_loss:8.4f}, MAE_E={error_e:8.3f} meV"
            )
    elif log_errors == "DipoleRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.4f}, RMSE_MU_per_atom={error_mu:8.2f} mDebye",
        )
    elif log_errors == "EnergyDipoleRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.4f}, RMSE_E_per_atom={error_e:8.3f} meV, RMSE_F={error_f:8.3f} meV / A, RMSE_Mu_per_atom={error_mu:8.2f} mDebye",
        )


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
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
    log_errors: str,
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
    logging.info("Loss metrics on validation set")
    epoch = start_epoch
    valid_loss = 0.0
    valid_loss, eval_metrics = evaluate(
        model=model,
        loss_fn=loss_fn,
        data_loader=valid_loader,
        output_args=output_args,
        device=device,
    )
    if start_epoch == 0:
        if (distributed and rank == 0) or not distributed:
            valid_err_log(
                valid_loss,
                eval_metrics,
                logger,
                log_errors,
                output_args["forces"],
                None,
            )
    else:
        if (distributed and rank == 0) or not distributed:
            valid_err_log(
                valid_loss,
                eval_metrics,
                logger,
                log_errors,
                output_args["forces"],
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
            loss_fn = swa.loss_fn
            swa.model.update_parameters(model)
            if epoch > start_epoch:
                swa.scheduler.step()

        # Train
        if distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            output_args=output_args,
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
                    loss_fn=loss_fn,
                    data_loader=valid_loader,
                    output_args=output_args,
                    device=device,
                )
                if (distributed and rank == 0) or not distributed:
                    valid_err_log(
                        valid_loss,
                        eval_metrics,
                        logger,
                        log_errors,
                        output_args["forces"],
                        epoch,
                    )
                    if log_wandb:
                        if output_args["forces"]:
                            wandb_log_dict = {
                                "epoch": epoch,
                                "valid_loss": valid_loss,
                                "valid_rmse_e_per_atom": eval_metrics[
                                    "rmse_e_per_atom"
                                ],
                                "valid_rmse_f": eval_metrics["rmse_f"],
                            }
                            wandb.log(wandb_log_dict)
                        else:
                            wandb_log_dict = {
                                "epoch": epoch,
                                "valid_loss": valid_loss,
                                "valid_rmse_e_per_atom": eval_metrics[
                                    "rmse_e_per_atom"
                                ],
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
            torch.distributed.broadcast(torch.tensor([epoch]), src=0)
            epoch = epoch.item()
        epoch += 1

    logging.info("Training complete")


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_args: Dict[str, bool],
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
            loss_fn=loss_fn,
            batch=batch,
            optimizer=optimizer,
            ema=ema,
            output_args=output_args,
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
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    world_size: int = 1,
    distributed: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    batch = batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    batch_dict = batch.to_dict()
    output = model(
        batch_dict,
        training=True,
        compute_force=output_args["forces"],
        compute_virials=output_args["virials"],
        compute_stress=output_args["stress"],
    )
    loss = loss_fn(pred=output, ref=batch)
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
) -> Tuple[float, Dict[str, Any]]:

    for param in model.parameters():
        param.requires_grad = False

    metrics = MACELoss(loss_fn=loss_fn).to(device)

    start_time = time.time()
    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        avg_loss, aux = metrics(batch, output)

    avg_loss, aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    for param in model.parameters():
        param.requires_grad = True

    return avg_loss, aux


class MACELoss(Metric):
    def __init__(self, loss_fn: torch.nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_data", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "stress_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_stress", default=[], dist_reduce_fx="cat")
        self.add_state(
            "virials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_virials", default=[], dist_reduce_fx="cat")
        self.add_state("delta_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Mus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus_per_atom", default=[], dist_reduce_fx="cat")

    def update(self, batch, output):  # pylint: disable=arguments-differ
        loss = self.loss_fn(pred=output, ref=batch)
        self.total_loss += loss
        self.num_data += batch.num_graphs

        if output.get("energy") is not None and batch.energy is not None:
            self.E_computed += 1.0
            self.delta_es.append(batch.energy - output["energy"])
            self.delta_es_per_atom.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
        if output.get("forces") is not None and batch.forces is not None:
            self.Fs_computed += 1.0
            self.fs.append(batch.forces)
            self.delta_fs.append(batch.forces - output["forces"])
        if output.get("stress") is not None and batch.stress is not None:
            self.stress_computed += 1.0
            self.delta_stress.append(batch.stress - output["stress"])
        if output.get("virials") is not None and batch.virials is not None:
            self.virials_computed += 1.0
            self.delta_virials.append(batch.virials - output["virials"])
            self.delta_virials_per_atom.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("dipole") is not None and batch.dipole is not None:
            self.Mus_computed += 1.0
            self.mus.append(batch.dipole)
            self.delta_mus.append(batch.dipole - output["dipole"])
            self.delta_mus_per_atom.append(
                (batch.dipole - output["dipole"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
            )

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return to_numpy(delta)

    def compute(self):
        aux = {}
        aux["loss"] = to_numpy(self.total_loss / self.num_data).item()
        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            aux["mae_e"] = compute_mae(delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e"] = compute_rmse(delta_es)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            aux["q95_e"] = compute_q95(delta_es)
        if self.Fs_computed:
            fs = self.convert(self.fs)
            delta_fs = self.convert(self.delta_fs)
            aux["mae_f"] = compute_mae(delta_fs)
            aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
            aux["rmse_f"] = compute_rmse(delta_fs)
            aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
            aux["q95_f"] = compute_q95(delta_fs)
        if self.stress_computed:
            delta_stress = self.convert(self.delta_stress)
            aux["mae_stress"] = compute_mae(delta_stress)
            aux["rmse_stress"] = compute_rmse(delta_stress)
            aux["q95_stress"] = compute_q95(delta_stress)
        if self.virials_computed:
            delta_virials = self.convert(self.delta_virials)
            delta_virials_per_atom = self.convert(self.delta_virials_per_atom)
            aux["mae_virials"] = compute_mae(delta_virials)
            aux["rmse_virials"] = compute_rmse(delta_virials)
            aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
            aux["q95_virials"] = compute_q95(delta_virials)
        if self.Mus_computed:
            mus = self.convert(self.mus)
            delta_mus = self.convert(self.delta_mus)
            delta_mus_per_atom = self.convert(self.delta_mus_per_atom)
            aux["mae_mu"] = compute_mae(delta_mus)
            aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
            aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
            aux["rmse_mu"] = compute_rmse(delta_mus)
            aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
            aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
            aux["q95_mu"] = compute_q95(delta_mus)

        return aux["loss"], aux
