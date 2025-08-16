###########################################################################################
# Training utils
# Authors: David Kovacs, Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import ast
import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from e3nn import o3
from prettytable import PrettyTable

from mace import data
from mace.tools import evaluate


@dataclasses.dataclass
class SubsetCollection:
    train: data.Configurations
    valid: data.Configurations
    tests: List[Tuple[str, data.Configurations]]


def get_dataset_from_xyz(
    log_dir: str,
    train_path: str,
    valid_path: str,
    valid_fraction: float,
    config_type_weights: Dict,
    test_path: str = None,
    seed: int = 1234,
    head_name: str = "Default",
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: str = "REF_stress",
    virials_key: str = "REF_virials",
    dipole_key: str = "REF_dipoles",
    charges_key: str = "REF_charges",
    head_key: str = "head",
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    """Load training and test dataset from xyz file"""
    atomic_energies_dict, all_train_configs = data.load_from_xyz(
        file_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        head_key=head_key,
        extract_atomic_energies=True,
        head_name=head_name,
    )
    logging.info(
        f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    )
    if valid_path is not None:
        _, valid_configs = data.load_from_xyz(
            file_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            head_key=head_key,
            extract_atomic_energies=False,
            head_name=head_name,
        )
        logging.info(
            f"Loaded {len(valid_configs)} validation configurations from '{valid_path}'"
        )
        train_configs = all_train_configs
    else:
        logging.info(
            "Using random %s%% of training set for validation", 100 * valid_fraction
        )
        train_configs, valid_configs = data.random_train_valid_split(
            all_train_configs, valid_fraction, seed, log_dir
        )

    test_configs = []
    if test_path is not None:
        _, all_test_configs = data.load_from_xyz(
            file_path=test_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            dipole_key=dipole_key,
            stress_key=stress_key,
            virials_key=virials_key,
            charges_key=charges_key,
            head_key=head_key,
            extract_atomic_energies=False,
            head_name=head_name,
        )
        # create list of tuples (config_type, list(Atoms))
        test_configs = data.test_config_types(all_test_configs)
        logging.info(
            f"Loaded {len(all_test_configs)} test configurations from '{test_path}'"
        )
    return (
        SubsetCollection(train=train_configs, valid=valid_configs, tests=test_configs),
        atomic_energies_dict,
    )


def get_config_type_weights(ct_weights):
    """
    Get config type weights from command line argument
    """
    try:
        config_type_weights = ast.literal_eval(ct_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}
    return config_type_weights


def extract_config_mace_model(model: torch.nn.Module) -> Dict[str, Any]:
    if model.__class__.__name__ != "ScaleShiftMACE":
        return {"error": "Model is not a ScaleShiftMACE model"}

    def radial_to_name(radial_type):
        if radial_type == "BesselBasis":
            return "bessel"
        if radial_type == "GaussianBasis":
            return "gaussian"
        if radial_type == "ChebychevBasis":
            return "chebyshev"
        return radial_type

    scale = model.scale_shift.scale
    shift = model.scale_shift.shift
    heads = model.heads if hasattr(model, "heads") else ["default"]
    model_mlp_irreps = (
        o3.Irreps(str(model.readouts[-1].hidden_irreps))
        if model.num_interactions.item() > 1
        else 1
    )
    mlp_irreps = o3.Irreps(f"{model_mlp_irreps.count((0, 1)) // len(heads)}x0e")
    try:
        correlation = (
            len(model.products[0].symmetric_contractions.contractions[0].weights) + 1
        )
    except AttributeError:
        correlation = model.products[0].symmetric_contractions.contraction_degree
    config = {
        "r_max": model.r_max.item(),
        "num_bessel": len(model.radial_embedding.bessel_fn.bessel_weights),
        "num_polynomial_cutoff": model.radial_embedding.cutoff_fn.p.item(),
        "max_ell": model.spherical_harmonics._lmax,  # pylint: disable=protected-access
        "interaction_cls": model.interactions[-1].__class__,
        "interaction_cls_first": model.interactions[0].__class__,
        "num_interactions": model.num_interactions.item(),
        "num_elements": len(model.atomic_numbers),
        "hidden_irreps": o3.Irreps(str(model.products[0].linear.irreps_out)),
        "MLP_irreps": (mlp_irreps if model.num_interactions.item() > 1 else 1),
        "gate": (
            model.readouts[-1]  # pylint: disable=protected-access
            .non_linearity._modules["acts"][0]
            .f
            if model.num_interactions.item() > 1
            else None
        ),
        "atomic_energies": model.atomic_energies_fn.atomic_energies.cpu().numpy(),
        "avg_num_neighbors": model.interactions[0].avg_num_neighbors,
        "atomic_numbers": model.atomic_numbers,
        "correlation": correlation,
        "radial_type": radial_to_name(
            model.radial_embedding.bessel_fn.__class__.__name__
        ),
        "radial_MLP": model.interactions[0].conv_tp_weights.hs[1:-1],
        "atomic_inter_scale": scale.cpu().numpy(),
        "atomic_inter_shift": shift.cpu().numpy(),
        "heads": heads,
    }
    return config


def extract_config_mace_committor_model(model: torch.nn.Module) -> Dict[str, Any]:
    if model.__class__.__name__ != "CommittorMACE":
        return {"error": "Model is not a CommittorMACE model"}

    def radial_to_name(radial_type):
        if radial_type == "BesselBasis":
            return "bessel"
        if radial_type == "GaussianBasis":
            return "gaussian"
        if radial_type == "ChebychevBasis":
            return "chebyshev"
        return radial_type

    p = model.parameteric_sigma.p
    c = model.parameteric_sigma.c
    trainable_c = model.trainable_c
    model_mlp_irreps = (
        o3.Irreps(str(model.readouts[-1].hidden_irreps))
        if model.num_interactions.item() > 1
        else 1
    )
    mlp_irreps = o3.Irreps(f"{model_mlp_irreps.count((0, 1))}x0e")
    try:
        correlation = (
            len(model.products[0].symmetric_contractions.contractions[0].weights) + 1
        )
    except AttributeError:
        correlation = model.products[0].symmetric_contractions.contraction_degree
    config = {
        "r_max": model.r_max.item(),
        "num_bessel": len(model.radial_embedding.bessel_fn.bessel_weights),
        "num_polynomial_cutoff": model.radial_embedding.cutoff_fn.p.item(),
        "max_ell": model.spherical_harmonics._lmax,  # pylint: disable=protected-access
        "interaction_cls": model.interactions[-1].__class__,
        "interaction_cls_first": model.interactions[0].__class__,
        "num_interactions": model.num_interactions.item(),
        "num_elements": len(model.atomic_numbers),
        "hidden_irreps": o3.Irreps(str(model.products[0].linear.irreps_out)),
        "MLP_irreps": (mlp_irreps if model.num_interactions.item() > 1 else 1),
        "gate": (
            model.readouts[-1]  # pylint: disable=protected-access
            .non_linearity._modules["acts"][0]
            .f
            if model.num_interactions.item() > 1
            else None
        ),
        "atomic_energies": model.atomic_energies_fn.atomic_energies.cpu().numpy(),
        "avg_num_neighbors": model.interactions[0].avg_num_neighbors,
        "atomic_numbers": model.atomic_numbers,
        "correlation": correlation,
        "radial_type": radial_to_name(
            model.radial_embedding.bessel_fn.__class__.__name__
        ),
        "radial_MLP": model.interactions[0].conv_tp_weights.hs[1:-1],
        "p": p.cpu().numpy(),
        "c": c.cpu().numpy(),
        "trainable_c": trainable_c,
    }
    return config


def extract_load(f: str, map_location: str = "cpu") -> torch.nn.Module:
    return extract_model(
        torch.load(f=f, map_location=map_location), map_location=map_location
    )


def remove_pt_head(
    model: torch.nn.Module, head_to_keep: Optional[str] = None
) -> torch.nn.Module:
    """Converts a multihead MACE model to a single head model by removing the pretraining head.

    Args:
        model (ScaleShiftMACE): The multihead MACE model to convert
        head_to_keep (Optional[str]): The name of the head to keep. If None, keeps the first non-PT head.

    Returns:
        ScaleShiftMACE: A new MACE model with only the specified head

    Raises:
        ValueError: If the model is not a multihead model or if the specified head is not found
    """
    if not hasattr(model, "heads") or len(model.heads) <= 1:
        raise ValueError("Model must be a multihead model with more than one head")

    # Get index of head to keep
    if head_to_keep is None:
        # Find first non-PT head
        try:
            head_idx = next(i for i, h in enumerate(model.heads) if h != "pt_head")
        except StopIteration as e:
            raise ValueError("No non-PT head found in model") from e
    else:
        try:
            head_idx = model.heads.index(head_to_keep)
        except ValueError as e:
            raise ValueError(f"Head {head_to_keep} not found in model") from e

    # Extract config and modify for single head
    model_config = extract_config_mace_model(model)
    model_config["heads"] = [model.heads[head_idx]]
    model_config["atomic_energies"] = (
        model.atomic_energies_fn.atomic_energies[head_idx]
        .unsqueeze(0)
        .detach()
        .cpu()
        .numpy()
    )
    model_config["atomic_inter_scale"] = model.scale_shift.scale[head_idx].item()
    model_config["atomic_inter_shift"] = model.scale_shift.shift[head_idx].item()
    mlp_count_irreps = model_config["MLP_irreps"].count((0, 1))
    # model_config["MLP_irreps"] = o3.Irreps(f"{mlp_count_irreps}x0e")

    new_model = model.__class__(**model_config)
    state_dict = model.state_dict()
    new_state_dict = {}

    for name, param in state_dict.items():
        if "atomic_energies" in name:
            new_state_dict[name] = param[head_idx : head_idx + 1]
        elif "scale" in name or "shift" in name:
            new_state_dict[name] = param[head_idx : head_idx + 1]
        elif "readouts" in name:
            channels_per_head = param.shape[0] // len(model.heads)
            start_idx = head_idx * channels_per_head
            end_idx = start_idx + channels_per_head
            if "linear_2.weight" in name:
                end_idx = start_idx + channels_per_head // 2
            # if (
            #     "readouts.0.linear.weight" in name
            #     or "readouts.1.linear_2.weight" in name
            # ):
            #     new_state_dict[name] = param[start_idx:end_idx] / (
            #         len(model.heads) ** 0.5
            #     )
            if "readouts.0.linear.weight" in name:
                new_state_dict[name] = param.reshape(-1, len(model.heads))[
                    :, head_idx
                ].flatten()
            elif "readouts.1.linear_1.weight" in name:
                new_state_dict[name] = param.reshape(
                    -1, len(model.heads), mlp_count_irreps
                )[:, head_idx, :].flatten()
            elif "readouts.1.linear_2.weight" in name:
                new_state_dict[name] = param.reshape(
                    len(model.heads), -1, len(model.heads)
                )[head_idx, :, head_idx].flatten() / (len(model.heads) ** 0.5)
            else:
                new_state_dict[name] = param[start_idx:end_idx]

        else:
            new_state_dict[name] = param

    # Load state dict into new model
    new_model.load_state_dict(new_state_dict)

    return new_model


def extract_model(model: torch.nn.Module, map_location: str = "cpu") -> torch.nn.Module:
    model_copy = model.__class__(**extract_config_mace_model(model))
    model_copy.load_state_dict(model.state_dict())
    return model_copy.to(map_location)


def dict_to_array(input_data, heads):
    if all(isinstance(value, np.ndarray) for value in input_data.values()):
        return np.array([input_data[head] for head in heads])
    if not all(isinstance(value, dict) for value in input_data.values()):
        return np.array([[input_data[head]] for head in heads])
    unique_keys = set()
    for inner_dict in input_data.values():
        unique_keys.update(inner_dict.keys())
    unique_keys = list(unique_keys)
    sorted_keys = sorted([int(key) for key in unique_keys])
    result_array = np.zeros((len(input_data), len(sorted_keys)))
    for _, (head_name, inner_dict) in enumerate(input_data.items()):
        for key, value in inner_dict.items():
            key_index = sorted_keys.index(int(key))
            head_index = heads.index(head_name)
            result_array[head_index][key_index] = value
    return result_array


class LRScheduler:
    def __init__(self, optimizer, args) -> None:
        self.scheduler = args.scheduler
        if args.scheduler == "ExponentialLR":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=args.lr_scheduler_gamma
            )
        elif args.scheduler == "ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=args.lr_factor,
                patience=args.scheduler_patience,
            )
        else:
            raise RuntimeError(f"Unknown scheduler: '{args.scheduler}'")

    def step(self, metrics=None, epoch=None):  # pylint: disable=E1123
        if self.scheduler == "ExponentialLR":
            self.lr_scheduler.step(epoch=epoch)
        elif self.scheduler == "ReduceLROnPlateau":
            self.lr_scheduler.step(  # pylint: disable=E1123
                metrics=metrics, epoch=epoch
            )

    def __getattr__(self, name):
        if name == "step":
            return self.step
        return getattr(self.lr_scheduler, name)


def custom_key(key):
    """
    Helper function to sort the keys of the data loader dictionary
    to ensure that the training set, and validation set
    are evaluated first
    """
    if key == "train":
        return (0, key)
    if key == "valid":
        return (1, key)
    return (2, key)


def create_error_table(
    table_type: str,
    all_data_loaders: dict,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    output_args: Dict[str, bool],
    log_wandb: bool,
    device: str,
    distributed: bool = False,
) -> PrettyTable:
    if log_wandb:
        import wandb
    table = PrettyTable()
    if table_type == "TotalRMSE":
        if output_args["forces"]:
            table.field_names = [
                "config_type",
                "RMSE E / meV",
                "RMSE F / meV / A",
                "relative F RMSE %",
            ]
        else:
            table.field_names = ["config_type", "RMSE E / meV"]
    elif table_type == "PerAtomRMSE":
        if output_args["forces"]:
            table.field_names = [
                "config_type",
                "RMSE E / meV / atom",
                "RMSE F / meV / A",
                "relative F RMSE %",
            ]
        else:
            table.field_names = ["config_type", "RMSE E / meV / atom"]
    elif table_type == "PerAtomRMSEstressvirials":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE Stress (Virials) / meV / A (A^3)",
        ]
    elif table_type == "TotalMAE":
        if output_args["forces"]:
            table.field_names = [
                "config_type",
                "MAE E / meV",
                "MAE F / meV / A",
                "relative F MAE %",
            ]
        else:
            table.field_names = ["config_type", "MAE E / meV"]
    elif table_type == "PerAtomMAE":
        if output_args["forces"]:
            table.field_names = [
                "config_type",
                "MAE E / meV / atom",
                "MAE F / meV / A",
                "relative F MAE %",
            ]
        else:
            table.field_names = ["config_type", "MAE E / meV / atom"]
    elif table_type == "DipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE MU / mDebye / atom",
            "relative MU RMSE %",
        ]
    elif table_type == "DipoleMAE":
        table.field_names = [
            "config_type",
            "MAE MU / mDebye / atom",
            "relative MU MAE %",
        ]
    elif table_type == "EnergyDipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "rel F RMSE %",
            "RMSE MU / mDebye / atom",
            "rel MU RMSE %",
        ]

    for name in sorted(all_data_loaders, key=custom_key):
        data_loader = all_data_loaders[name]
        logging.info(f"Evaluating {name} ...")
        _, metrics = evaluate(
            model,
            loss_fn=loss_fn,
            data_loader=data_loader,
            output_args=output_args,
            device=device,
        )
        if distributed:
            torch.distributed.barrier()

        del data_loader
        torch.cuda.empty_cache()
        if log_wandb:
            wandb_log_dict = {
                name
                + "_final_rmse_e_per_atom": metrics["rmse_e_per_atom"]
                * 1e3,  # meV / atom
                name + "_final_rmse_f": metrics["rmse_f"] * 1e3,  # meV / A
                name + "_final_rel_rmse_f": metrics["rel_rmse_f"],
            }
            wandb.log(wandb_log_dict)
        if table_type == "TotalRMSE":
            if output_args["forces"]:
                table.add_row(
                    [
                        name,
                        f"{metrics['rmse_e'] * 1000:8.3f}",
                        f"{metrics['rmse_f'] * 1000:8.3f}",
                        f"{metrics['rel_rmse_f']:8.3f}",
                    ]
                )
            else:
                table.add_row([name, f"{metrics['rmse_e'] * 1000:8.3f}"])
        elif table_type == "PerAtomRMSE":
            if output_args["forces"]:
                table.add_row(
                    [
                        name,
                        f"{metrics['rmse_e_per_atom'] * 1000:8.3f}",
                        f"{metrics['rmse_f'] * 1000:8.3f}",
                        f"{metrics['rel_rmse_f']:8.3f}",
                    ]
                )
            else:
                table.add_row([name, f"{metrics['rmse_e_per_atom'] * 1000:8.3f}"])
        elif (
            table_type == "PerAtomRMSEstressvirials"
            and metrics["rmse_stress"] is not None
        ):
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:8.3f}",
                    f"{metrics['rmse_f'] * 1000:8.3f}",
                    f"{metrics['rel_rmse_f']:8.3f}",
                    f"{metrics['rmse_stress'] * 1000:8.3f}",
                ]
            )
        elif (
            table_type == "PerAtomRMSEstressvirials"
            and metrics["rmse_virials"] is not None
        ):
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:8.3f}",
                    f"{metrics['rmse_f'] * 1000:8.3f}",
                    f"{metrics['rel_rmse_f']:8.3f}",
                    f"{metrics['rmse_virials'] * 1000:8.3f}",
                ]
            )
        elif table_type == "TotalMAE":
            if output_args["forces"]:
                table.add_row(
                    [
                        name,
                        f"{metrics['mae_e'] * 1000:8.3f}",
                        f"{metrics['mae_f'] * 1000:8.3f}",
                        f"{metrics['rel_mae_f']:8.3f}",
                    ]
                )
            else:
                table.add_row([name, f"{metrics['mae_e'] * 1000:8.3f}"])
        elif table_type == "PerAtomMAE":
            if output_args["forces"]:
                table.add_row(
                    [
                        name,
                        f"{metrics['mae_e_per_atom'] * 1000:8.3f}",
                        f"{metrics['mae_f'] * 1000:8.3f}",
                        f"{metrics['rel_mae_f']:8.3f}",
                    ]
                )
            else:
                table.add_row([name, f"{metrics['mae_e_per_atom'] * 1000:8.3f}"])
        elif table_type == "DipoleRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_mu_per_atom'] * 1000:8.3f}",
                    f"{metrics['rel_rmse_mu']:8.3f}",
                ]
            )
        elif table_type == "DipoleMAE":
            table.add_row(
                [
                    name,
                    f"{metrics['mae_mu_per_atom'] * 1000:8.3f}",
                    f"{metrics['rel_mae_mu']:8.3f}",
                ]
            )
        elif table_type == "EnergyDipoleRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:8.3f}",
                    f"{metrics['rmse_f'] * 1000:8.3f}",
                    f"{metrics['rel_rmse_f']:8.3f}",
                    f"{metrics['rmse_mu_per_atom'] * 1000:8.3f}",
                    f"{metrics['rel_rmse_mu']:8.3f}",
                ]
            )
    return table
