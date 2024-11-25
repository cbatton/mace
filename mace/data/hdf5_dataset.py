import h5py
import torch
from torch.utils.data import Dataset

from mace.data.atomic_data import AtomicData


class HDF5Dataset(Dataset):
    def __init__(self, file_path, indices):
        super(HDF5Dataset, self).__init__()  # pylint: disable=super-with-arguments
        self.file_path = file_path
        self.indices = indices
        self._file = None

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

        return atomic_data
