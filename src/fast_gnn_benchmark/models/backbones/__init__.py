import torch

from fast_gnn_benchmark.models.backbones.gnn import load_gnn
from fast_gnn_benchmark.models.backbones.mlp import MLP
from fast_gnn_benchmark.models.backbones.sgformer import SGFormer
from fast_gnn_benchmark.schemas.model import (
    ArchitectureParametersChoices,
    ArchitectureType,
    GNNParameters,
    MLPParameters,
    SGCParameters,
    SGFormerParameters,
)


def load_backbone(architecture_parameters: ArchitectureParametersChoices) -> torch.nn.Module:
    match architecture_parameters.architecture_type:
        case (
            ArchitectureType.GCN
            | ArchitectureType.SAGE
            | ArchitectureType.GAT
            | ArchitectureType.SGC
            | ArchitectureType.SGC2
        ):
            assert isinstance(architecture_parameters, GNNParameters | SGCParameters)
            return load_gnn(architecture_parameters)

        case ArchitectureType.SGFORMER:
            assert isinstance(architecture_parameters, SGFormerParameters)
            return SGFormer(architecture_parameters)

        case ArchitectureType.MLP:
            assert isinstance(architecture_parameters, MLPParameters)
            return MLP(architecture_parameters)

        case _:
            raise ValueError(f"Invalid architecture type: {architecture_parameters.architecture_type}")
