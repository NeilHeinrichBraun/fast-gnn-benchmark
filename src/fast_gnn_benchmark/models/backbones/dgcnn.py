import torch
from torch.nn import Conv1d, MaxPool1d, ModuleList
from torch_geometric.nn import MLP, GATConv, GCNConv, SAGEConv, SortAggregation

from fast_gnn_benchmark.schemas.model import ArchitectureType, DGCNNParameters

_INNER_GNN_CONV_CLASSES: dict[ArchitectureType, type[torch.nn.Module]] = {
    ArchitectureType.GCN: GCNConv,
    ArchitectureType.SAGE: SAGEConv,
    ArchitectureType.GAT: GATConv,
}


class DGCNN(torch.nn.Module):
    """DGCNN backbone for SEAL (Zhang & Chen, 2018).

    Note: `num_layers` counts the hidden-dim convs only — an implicit
    final 1-channel conv is appended, so `num_layers=3` builds 4 convs total.
    """

    def __init__(self, parameters: DGCNNParameters):
        super().__init__()

        ConvClass = _INNER_GNN_CONV_CLASSES[parameters.inner_gnn_type]
        k = parameters.k

        self.convs = ModuleList()
        # First conv is lazy: it infers in_channels from the first batch's x.shape[-1].
        # Removes the need to know max_z + feature_dim upfront in the YAML.
        self.convs.append(ConvClass(-1, parameters.hidden_dim))
        for _ in range(parameters.num_layers - 1):
            self.convs.append(ConvClass(parameters.hidden_dim, parameters.hidden_dim))
        self.convs.append(ConvClass(parameters.hidden_dim, 1))

        total_latent_dim = parameters.hidden_dim * parameters.num_layers + 1

        self.pool = SortAggregation(k)
        self.conv1 = Conv1d(1, parameters.conv1d_channels[0], total_latent_dim, total_latent_dim)
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(
            parameters.conv1d_channels[0],
            parameters.conv1d_channels[1],
            parameters.conv1d_kernel_size,
            1,
        )

        dense_dim = int((k - 2) / 2 + 1)
        dense_dim = (dense_dim - parameters.conv1d_kernel_size + 1) * parameters.conv1d_channels[1]

        self.mlp = MLP(
            [dense_dim, parameters.mlp_hidden_dim, 1],
            dropout=parameters.dropout,
            norm=None,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        xs = [x]
        for conv in self.convs:
            xs.append(conv(xs[-1], edge_index).tanh())
        x = torch.cat(xs[1:], dim=-1)

        x = self.pool(x, batch)
        x = x.unsqueeze(1)
        x = self.conv1(x).relu()
        x = self.maxpool1d(x)
        x = self.conv2(x).relu()
        x = x.view(x.size(0), -1)

        return self.mlp(x)
