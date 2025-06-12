###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.tools.scatter import scatter_sum

from .blocks import (
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
)
from .models import MACE
from .utils import get_edge_vectors_and_lengths

# pylint: disable=C0302


@compile_mode("script")
class ParametricSigmoid(torch.nn.Module):
    def __init__(self, p: float = 3.0, c: float = 0.0, trainable_c: bool = False):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.trainable_c = trainable_c
        if self.trainable_c:
            self.c = torch.nn.Parameter(
                torch.tensor(c, dtype=torch.get_default_dtype())
            )
        else:
            self.register_buffer("c", torch.tensor(c, dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-self.p * x + self.c))

    def update_c(self, new_c: float) -> None:
        self.c.data.fill_(new_c)


@compile_mode("script")
class CommittorMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        cueq_config: Optional[Dict[str, Any]] = None,
        p: float = 3.0,
        trainable_c: bool = False,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps, cueq_config=cueq_config))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate, cueq_config=cueq_config)
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps), cueq_config=cueq_config)
        # pass through a modified sigmoid
        self.psigmoid = ParametricSigmoid(p=p, trainable_c=trainable_c)

    def read_MACE_model(
        self,
        model: MACE,
        disable_grad: bool = False,
    ) -> None:
        # Load in the relevant parameters of a model
        # trained on energies and forces
        self.node_embedding.load_state_dict(model.node_embedding.state_dict())
        self.interactions.load_state_dict(model.interactions.state_dict())
        self.products.load_state_dict(model.products.state_dict())
        # Disable requires_grad for only the loaded parameters
        if disable_grad:
            for param in self.node_embedding.parameters():
                param.requires_grad_(False)
            for param in self.interactions.parameters():
                param.requires_grad_(False)
            for param in self.products.parameters():
                param.requires_grad_(False)

    def disable_grad_readout(self) -> None:
        # Disable requires_grad for the readout
        for param in self.readouts.parameters():
            param.requires_grad_(False)
        if self.psigmoid.trainable_c:
            self.psigmoid.c.requires_grad_(False)

    def enable_grad_readout(self) -> None:
        # Enable requires_grad for the readout
        for param in self.readouts.parameters():
            param.requires_grad_(True)
        if self.psigmoid.trainable_c:
            self.psigmoid.c.requires_grad_(True)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        contrib = []
        node_contrib_list = []
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_feats_list.append(node_feats)
            node_contrib = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_contrib, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            contrib.append(energy)
            node_contrib_list.append(node_contrib)

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(contrib, dim=-1)
        total_contributions = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_contrib_contributions = torch.stack(node_contrib_list, dim=-1)
        node_contrib = torch.sum(node_contrib_contributions, dim=-1)  # [n_nodes, ]

        # Pass through sigmoid
        committors = self.psigmoid(total_contributions)

        return {
            "committor": committors,
            "total_contributions": total_contributions,
            "node_contrib": node_contrib,
            "contributions": contributions,
            "node_feats": node_feats_out,
        }
