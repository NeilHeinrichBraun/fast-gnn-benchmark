import torch
import torch.nn.functional as F

from fast_gnn_benchmark.schemas.model import MLPParameters


class MLP(torch.nn.Module):
    def __init__(self, architecture_parameters: MLPParameters):
        super().__init__()
        self.architecture_parameters = architecture_parameters

        self.projection_layers = torch.nn.ModuleList()
        if architecture_parameters.use_layer_norm:
            self.layer_norms = torch.nn.ModuleList()

        for layer_index in range(architecture_parameters.num_layers):
            if layer_index == 0:
                input_dim = architecture_parameters.input_dim
            else:
                input_dim = architecture_parameters.hidden_dim

            if layer_index == architecture_parameters.num_layers - 1:
                output_dim = architecture_parameters.output_dim
            else:
                output_dim = architecture_parameters.hidden_dim

            self.projection_layers.append(torch.nn.Linear(input_dim, output_dim))
            if architecture_parameters.use_layer_norm:
                self.layer_norms.append(torch.nn.LayerNorm(output_dim))

    def forward(self, x, edge_index) -> torch.Tensor:  # noqa: ARG002
        for layer_index in range(self.architecture_parameters.num_layers):
            projected_x = self.projection_layers[layer_index](x)

            if layer_index != self.architecture_parameters.num_layers - 1:
                if self.architecture_parameters.use_layer_norm:
                    projected_x = self.layer_norms[layer_index](projected_x)
                projected_x = F.relu(projected_x)
                projected_x = F.dropout(projected_x, p=self.architecture_parameters.dropout, training=self.training)

            if (
                self.architecture_parameters.use_residual
                and layer_index != 0
                and layer_index != self.architecture_parameters.num_layers - 1
            ):
                x = x + projected_x
            else:
                x = projected_x

        return x

