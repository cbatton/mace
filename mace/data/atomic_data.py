###########################################################################################
# Atomic Data Class for handling molecules as graphs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import List, Optional, Sequence

import h5py
import torch
import torch.utils.data

from mace.tools import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    to_one_hot,
    torch_geometric,
    voigt_to_matrix,
)

from .neighborhood import get_neighborhood
from .utils import Configuration, Configurations


class AtomicData(torch_geometric.data.Data):
    num_graphs: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    cell: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    stress: torch.Tensor
    virials: torch.Tensor
    dipole: torch.Tensor
    charges: torch.Tensor
    weight: torch.Tensor
    energy_weight: torch.Tensor
    forces_weight: torch.Tensor
    stress_weight: torch.Tensor
    virials_weight: torch.Tensor

    def __init__(
        self,
        edge_index: torch.Tensor,  # [2, n_edges]
        node_attrs: torch.Tensor,  # [n_nodes, n_node_feats]
        positions: torch.Tensor,  # [n_nodes, 3]
        shifts: torch.Tensor,  # [n_edges, 3],
        unit_shifts: torch.Tensor,  # [n_edges, 3]
        cell: Optional[torch.Tensor],  # [3,3]
        weight: Optional[torch.Tensor],  # [,]
        energy_weight: Optional[torch.Tensor],  # [,]
        forces_weight: Optional[torch.Tensor],  # [,]
        stress_weight: Optional[torch.Tensor],  # [,]
        virials_weight: Optional[torch.Tensor],  # [,]
        forces: Optional[torch.Tensor],  # [n_nodes, 3]
        energy: Optional[torch.Tensor],  # [, ]
        stress: Optional[torch.Tensor],  # [1,3,3]
        virials: Optional[torch.Tensor],  # [1,3,3]
        dipole: Optional[torch.Tensor],  # [, 3]
        charges: Optional[torch.Tensor],  # [n_nodes, ]
    ):
        # Check shapes
        num_nodes = node_attrs.shape[0]

        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert unit_shifts.shape[1] == 3
        assert len(node_attrs.shape) == 2
        assert weight is None or len(weight.shape) == 0
        assert energy_weight is None or len(energy_weight.shape) == 0
        assert forces_weight is None or len(forces_weight.shape) == 0
        assert stress_weight is None or len(stress_weight.shape) == 0
        assert virials_weight is None or len(virials_weight.shape) == 0
        assert cell is None or cell.shape == (3, 3)
        assert forces is None or forces.shape == (num_nodes, 3)
        assert energy is None or len(energy.shape) == 0
        assert stress is None or stress.shape == (1, 3, 3)
        assert virials is None or virials.shape == (1, 3, 3)
        assert dipole is None or dipole.shape[-1] == 3
        assert charges is None or charges.shape == (num_nodes,)
        # Aggregate data
        data = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "positions": positions,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "node_attrs": node_attrs,
            "weight": weight,
            "energy_weight": energy_weight,
            "forces_weight": forces_weight,
            "stress_weight": stress_weight,
            "virials_weight": virials_weight,
            "forces": forces,
            "energy": energy,
            "stress": stress,
            "virials": virials,
            "dipole": dipole,
            "charges": charges,
        }
        super().__init__(**data)

    @classmethod
    def from_config(
        cls, config: Configuration, z_table: AtomicNumberTable, cutoff: float
    ) -> "AtomicData":
        edge_index, shifts, unit_shifts = get_neighborhood(
            positions=config.positions, cutoff=cutoff, pbc=config.pbc, cell=config.cell
        )
        indices = atomic_numbers_to_indices(config.atomic_numbers, z_table=z_table)
        one_hot = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )

        cell = (
            torch.tensor(config.cell, dtype=torch.get_default_dtype())
            if config.cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )

        weight = (
            torch.tensor(config.weight, dtype=torch.get_default_dtype())
            if config.weight is not None
            else 1
        )

        energy_weight = (
            torch.tensor(config.energy_weight, dtype=torch.get_default_dtype())
            if config.energy_weight is not None
            else 1
        )

        forces_weight = (
            torch.tensor(config.forces_weight, dtype=torch.get_default_dtype())
            if config.forces_weight is not None
            else 1
        )

        stress_weight = (
            torch.tensor(config.stress_weight, dtype=torch.get_default_dtype())
            if config.stress_weight is not None
            else 1
        )

        virials_weight = (
            torch.tensor(config.virials_weight, dtype=torch.get_default_dtype())
            if config.virials_weight is not None
            else 1
        )

        forces = (
            torch.tensor(config.forces, dtype=torch.get_default_dtype())
            if config.forces is not None
            else None
        )
        energy = (
            torch.tensor(config.energy, dtype=torch.get_default_dtype())
            if config.energy is not None
            else None
        )
        stress = (
            voigt_to_matrix(
                torch.tensor(config.stress, dtype=torch.get_default_dtype())
            ).unsqueeze(0)
            if config.stress is not None
            else None
        )
        virials = (
            voigt_to_matrix(
                torch.tensor(config.virials, dtype=torch.get_default_dtype())
            ).unsqueeze(0)
            if config.virials is not None
            else None
        )
        dipole = (
            torch.tensor(config.dipole, dtype=torch.get_default_dtype()).unsqueeze(0)
            if config.dipole is not None
            else None
        )
        charges = (
            torch.tensor(config.charges, dtype=torch.get_default_dtype())
            if config.charges is not None
            else None
        )

        return cls(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            positions=torch.tensor(config.positions, dtype=torch.get_default_dtype()),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
            cell=cell,
            node_attrs=one_hot,
            weight=weight,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            stress_weight=stress_weight,
            virials_weight=virials_weight,
            forces=forces,
            energy=energy,
            stress=stress,
            virials=virials,
            dipole=dipole,
            charges=charges,
        )


def get_data_loader(
    dataset: Sequence[AtomicData],
    batch_size: int,
    shuffle=True,
    drop_last=False,
) -> torch.utils.data.DataLoader:
    return torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def save_dataset_as_HDF5(
    dataset: List, out_name: str, compression: str = "gzip", compression_level: int = 4
) -> None:
    with h5py.File(out_name, "w") as f:
        for i, data in enumerate(dataset):
            grp = f.create_group(f"config_{i}")
            grp.create_dataset("num_nodes", data=data.num_nodes, dtype="i8")
            grp.create_dataset(
                "edge_index",
                data=data.edge_index,
                compression=compression,
                compression_opts=compression_level,
                dtype="i8",
            )
            grp.create_dataset(
                "positions",
                data=data.positions,
                compression=compression,
                compression_opts=compression_level,
                dtype="f8",
            )
            grp.create_dataset(
                "shifts",
                data=data.shifts,
                compression=compression,
                compression_opts=compression_level,
                dtype="f8",
            )
            grp.create_dataset(
                "unit_shifts",
                data=data.unit_shifts,
                compression=compression,
                compression_opts=compression_level,
                dtype="f8",
            )
            grp.create_dataset(
                "cell",
                data=data.cell,
                compression=compression,
                compression_opts=compression_level,
                dtype="f8",
            )
            grp.create_dataset(
                "node_attrs",
                data=data.node_attrs,
                compression=compression,
                compression_opts=compression_level,
                dtype="f8",
            )
            grp.create_dataset("weight", data=data.weight, dtype="f8")
            grp.create_dataset("energy_weight", data=data.energy_weight, dtype="f8")
            grp.create_dataset("forces_weight", data=data.forces_weight, dtype="f8")
            grp.create_dataset("stress_weight", data=data.stress_weight, dtype="f8")
            grp.create_dataset("virials_weight", data=data.virials_weight, dtype="f8")
            grp.create_dataset(
                "forces",
                data=data.forces,
                compression=compression,
                compression_opts=compression_level,
                dtype="f8",
            )
            grp.create_dataset("energy", data=data.energy, dtype="f8")
            grp.create_dataset(
                "stress",
                data=data.stress,
                compression=compression,
                compression_opts=compression_level,
                dtype="f8",
            )
            grp.create_dataset(
                "virials",
                data=data.virials,
                compression=compression,
                compression_opts=compression_level,
                dtype="f8",
            )
            grp.create_dataset(
                "charges",
                data=data.charges,
                compression=compression,
                compression_opts=compression_level,
                dtype="f8",
            )
            try:
                grp.create_dataset(
                    "dipole",
                    data=data.dipole,
                    compression=compression,
                    compression_opts=compression_level,
                    dtype="f8",
                )
            except TypeError:
                pass


def load_dataset_from_HDF5(file_path: str) -> List[AtomicData]:
    dataset = []
    with h5py.File(file_path, "r") as f:
        # Iterate through the groups in the HDF5 file
        for config_key in f.keys():  # pylint: disable=C0206
            grp = f[config_key]

            # Check for the existence of the "dipole" key in the group
            dipole = (
                torch.tensor(grp["dipole"][()], dtype=torch.get_default_dtype())
                if "dipole" in grp
                else None
            )

            atomic_data = AtomicData(
                edge_index=torch.tensor(
                    grp["edge_index"][()], dtype=torch.long
                ),  # [2, n_edges]
                node_attrs=torch.tensor(
                    grp["node_attrs"][()], dtype=torch.get_default_dtype()
                ),  # [n_nodes, n_node_feats]
                positions=torch.tensor(
                    grp["positions"][()], dtype=torch.get_default_dtype()
                ),  # [n_nodes, 3]
                shifts=torch.tensor(
                    grp["shifts"][()], dtype=torch.get_default_dtype()
                ),  # [n_edges, 3]
                unit_shifts=torch.tensor(
                    grp["unit_shifts"][()], dtype=torch.get_default_dtype()
                ),  # [n_edges, 3]
                cell=torch.tensor(
                    grp["cell"][()], dtype=torch.get_default_dtype()
                ),  # [3, 3]
                weight=torch.tensor(
                    grp["weight"][()], dtype=torch.get_default_dtype()
                ),  # [,]
                energy_weight=torch.tensor(
                    grp["energy_weight"][()], dtype=torch.get_default_dtype()
                ),  # [,]
                forces_weight=torch.tensor(
                    grp["forces_weight"][()], dtype=torch.get_default_dtype()
                ),  # [,]
                stress_weight=torch.tensor(
                    grp["stress_weight"][()], dtype=torch.get_default_dtype()
                ),  # [,]
                virials_weight=torch.tensor(
                    grp["virials_weight"][()], dtype=torch.get_default_dtype()
                ),  # [,]
                forces=torch.tensor(
                    grp["forces"][()], dtype=torch.get_default_dtype()
                ),  # [n_nodes, 3]
                energy=torch.tensor(
                    grp["energy"][()], dtype=torch.get_default_dtype()
                ),  # [,]
                stress=torch.tensor(
                    grp["stress"][()], dtype=torch.get_default_dtype()
                ),  # [1, 3, 3]
                virials=torch.tensor(
                    grp["virials"][()], dtype=torch.get_default_dtype()
                ),  # [1, 3, 3]
                dipole=dipole,  # [3,] or None if not present
                charges=torch.tensor(
                    grp["charges"][()], dtype=torch.get_default_dtype()
                ),  # [n_nodes,]
            )

            # Append to the list of dataset
            dataset.append(atomic_data)

    return dataset


def save_AtomicData_to_HDF5(data, i, h5_file) -> None:
    grp = h5_file.create_group(f"config_{i}")
    grp["num_nodes"] = data.num_nodes
    grp["edge_index"] = data.edge_index
    grp["positions"] = data.positions
    grp["shifts"] = data.shifts
    grp["unit_shifts"] = data.unit_shifts
    grp["cell"] = data.cell
    grp["node_attrs"] = data.node_attrs
    grp["weight"] = data.weight
    grp["energy_weight"] = data.energy_weight
    grp["forces_weight"] = data.forces_weight
    grp["stress_weight"] = data.stress_weight
    grp["virials_weight"] = data.virials_weight
    grp["forces"] = data.forces
    grp["energy"] = data.energy
    grp["stress"] = data.stress
    grp["virials"] = data.virials
    grp["dipole"] = data.dipole
    grp["charges"] = data.charges


def save_configurations_as_HDF5(configurations: Configurations, _, h5_file) -> None:
    grp = h5_file.create_group("config_batch_0")
    for j, config in enumerate(configurations):
        subgroup_name = f"config_{j}"
        subgroup = grp.create_group(subgroup_name)
        subgroup["atomic_numbers"] = write_value(config.atomic_numbers)
        subgroup["positions"] = write_value(config.positions)
        subgroup["energy"] = write_value(config.energy)
        subgroup["forces"] = write_value(config.forces)
        subgroup["stress"] = write_value(config.stress)
        subgroup["virials"] = write_value(config.virials)
        subgroup["dipole"] = write_value(config.dipole)
        subgroup["charges"] = write_value(config.charges)
        subgroup["cell"] = write_value(config.cell)
        subgroup["pbc"] = write_value(config.pbc)
        subgroup["weight"] = write_value(config.weight)
        subgroup["energy_weight"] = write_value(config.energy_weight)
        subgroup["forces_weight"] = write_value(config.forces_weight)
        subgroup["stress_weight"] = write_value(config.stress_weight)
        subgroup["virials_weight"] = write_value(config.virials_weight)
        subgroup["config_type"] = write_value(config.config_type)


def write_value(value):
    return value if value is not None else "None"
