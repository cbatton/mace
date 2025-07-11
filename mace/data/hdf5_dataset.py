import re
from typing import List

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from mace.data.atomic_data import AtomicData
from mace.data.utils import Configuration


def natural_sort(l):
    convert = lambda text: (  # pylint: disable=C3001
        int(text) if text.isdigit() else text.lower()
    )
    alphanum_key = lambda key: [  # pylint: disable=C3001
        convert(c) for c in re.split("([0-9]+)", key)
    ]
    return sorted(l, key=alphanum_key)


class HDF5Dataset(Dataset):
    def __init__(self, file_path, indices, weight=None):
        super(HDF5Dataset, self).__init__()  # pylint: disable=super-with-arguments
        self.file_path = file_path
        self.indices = indices
        self._file = None
        self.weight = weight

    @property
    def file(self):
        if self._file is None:
            # If a file has not already been opened, open one here
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def __getstate__(self):
        _d = dict(self.__dict__)

        # An opened h5py.File cannot be pickled, so we must exclude it from the state
        _d["_file"] = None
        return _d

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # compute the index of the batch
        index = self.indices[index]
        grp = self.file["config_" + str(index)]

        # check for the existense of the "dipole" key in the group
        dipole = (
            torch.tensor(grp["dipole"][()], dtype=torch.get_default_dtype())
            if "dipole" in grp
            else None
        )
        # Use self.weight if it is provided, otherwise get from file
        weight = (
            torch.tensor(self.weight, dtype=torch.get_default_dtype())
            if self.weight is not None
            else torch.tensor(grp["weight"][()], dtype=torch.get_default_dtype())
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
            weight=weight,  # [,]
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
            head=torch.tensor(grp["head"][()], dtype=torch.long),  # [,]
        )

        return atomic_data


class HDF5DatasetCommittorTrain(Dataset):
    def __init__(self, file_path, r_max, z_table, indices):
        super(  # pylint: disable=super-with-arguments
            HDF5DatasetCommittorTrain, self
        ).__init__()
        self.file_path = file_path
        self.r_max = r_max
        self.z_table = z_table
        self._file = None
        self.keys = self.file.keys()
        self.keys = list(self.keys)
        self.indices = indices

    @property
    def file(self):
        if self._file is None:
            # If a file has not already been opened, open one here
            self._file = h5py.File(self.file_path, "r")
            #
        return self._file

    def __getstate__(self):
        _d = dict(self.__dict__)

        # An opened h5py.File cannot be pickled, so we must exclude it from the state
        _d["_file"] = None
        return _d

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # compute the index of the batch
        index = self.keys[self.indices[index]]
        grp = self.file[index]

        # unpack the group
        dipole = (
            torch.tensor(grp["dipole"][()], dtype=torch.get_default_dtype())
            if "dipole" in grp
            else None
        )
        atomic_numbers = np.vectorize(self.z_table.index_to_z)(
            grp["node_attrs"][()].argmax(axis=1).astype(int)
        )
        positions = grp["positions"][()]
        energy = unpack_value(grp["energy"][()])
        forces = unpack_value(grp["forces"][()])
        stress = unpack_value(grp["stress"][()])
        virials = unpack_value(grp["virials"][()])
        charges = unpack_value(grp["charges"][()])
        head = unpack_value(grp["head"][()])
        weight = unpack_value(grp["weight"][()])
        energy_weight = unpack_value(grp["energy_weight"][()])
        forces_weight = unpack_value(grp["forces_weight"][()])
        stress_weight = unpack_value(grp["stress_weight"][()])
        virials_weight = unpack_value(grp["virials_weight"][()])
        config_type = "Default"
        pbc = [True, True, True]
        cell = unpack_value(grp["cell"][()])

        # create the atomic data object for the main configuration
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
            head=torch.tensor(grp["head"][()], dtype=torch.long),  # [,]
        )

        # unpack the CV data
        cv_data = []
        cv_keys = natural_sort(grp["cv_dt"].keys())
        for key in cv_keys:
            cv_data.append(grp["cv_dt"][key][()])
        cv_data = torch.tensor(cv_data, dtype=torch.get_default_dtype())

        # unpack the additional configurations
        atomic_data_dt_list = []
        for key in natural_sort(grp["positions_dt"].keys()):
            positions = grp["positions_dt"][key][()]
            config_dt = Configuration(
                atomic_numbers=atomic_numbers,
                positions=positions,
                energy=energy,
                forces=forces,
                stress=stress[0],
                virials=virials[0],
                dipole=dipole,
                charges=charges,
                head=head,
                weight=weight,
                energy_weight=energy_weight,
                forces_weight=forces_weight,
                stress_weight=stress_weight,
                virials_weight=virials_weight,
                config_type=config_type,
                pbc=pbc,
                cell=cell,
            )
            atomic_data_dt = AtomicData.from_config(
                config_dt,
                z_table=self.z_table,
                cutoff=self.r_max,
                head_index=head,
            )
            atomic_data_dt_list.append(atomic_data_dt)
        return atomic_data, cv_data, atomic_data_dt_list


class HDF5DatasetCommittorValid(Dataset):
    def __init__(self, file_path, indices):
        super(  # pylint: disable=super-with-arguments
            HDF5DatasetCommittorValid, self
        ).__init__()
        self.file_path = file_path
        self._file = None
        self.keys = self.file.keys()
        self.keys = list(self.keys)
        self.indices = indices

    @property
    def file(self):
        if self._file is None:
            # If a file has not already been opened, open one here
            self._file = h5py.File(self.file_path, "r")
            #
        return self._file

    def __getstate__(self):
        _d = dict(self.__dict__)

        # An opened h5py.File cannot be pickled, so we must exclude it from the state
        _d["_file"] = None
        return _d

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # compute the index of the batch
        index = self.keys[self.indices[index]]
        grp = self.file[index]

        # unpack the group
        dipole = (
            torch.tensor(grp["dipole"][()], dtype=torch.get_default_dtype())
            if "dipole" in grp
            else None
        )

        # create the atomic data object for the main configuration
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
            head=torch.tensor(grp["head"][()], dtype=torch.long),  # [,]
        )

        # get the committor value
        committor = torch.tensor(grp["committor"][()], dtype=torch.get_default_dtype())

        return atomic_data, committor


def dataset_from_sharded_hdf5(files: List[str], indices: List[torch.Tensor]):
    datasets = []
    for file in files:
        datasets.append(HDF5Dataset(file, indices))
    return ConcatDataset(datasets)


def unpack_value(value):
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    return None if str(value) == "None" else value
