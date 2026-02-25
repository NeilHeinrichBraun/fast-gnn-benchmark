import torch

from fast_gnn_benchmark.schemas.model import EmbedderParameters


def load_embedder(embedder_parameters: EmbedderParameters) -> torch.nn.Module:
    return Embedder(
        use_embedding=embedder_parameters.use_embedding,
        embedding_dim=embedder_parameters.embedding_dim,
        num_nodes=embedder_parameters.num_nodes,
        embedding_only=embedder_parameters.embedding_only,
        initializer=embedder_parameters.initializer,
    )


class Embedder(torch.nn.Module):
    def __init__(
        self,
        use_embedding: bool,
        embedding_dim: int,
        num_nodes: int,
        embedding_only: bool = False,
        initializer: str = "orthogonal",
    ):
        """
        Args:
            embedding_dim: The dimension of the embedding.
            num_nodes: The number of nodes in the graph.
            embedding_only: Whether to only return the embedding or the embedding concatenated with the features of the data.
            initializer: The initializer to use for the embedding layer. Choices are "orthogonal", "uniform", "ones", "normal".
        """
        super().__init__()
        self.use_embedding = use_embedding
        if use_embedding:
            self.embedding_dim = embedding_dim
            self.embedding_only = embedding_only
            self.num_nodes = num_nodes
            self.initializer = initializer

            self.embedding_layer = torch.nn.Embedding(num_nodes, embedding_dim)
            match initializer:
                case "orthogonal":
                    torch.nn.init.orthogonal_(self.embedding_layer.weight)
                case "uniform":
                    torch.nn.init.uniform_(self.embedding_layer.weight)
                case "ones":
                    torch.nn.init.ones_(self.embedding_layer.weight)
                case "normal":
                    torch.nn.init.normal_(self.embedding_layer.weight)
                case _:
                    raise ValueError(
                        f"Invalid initializer: {initializer}. Choices are 'orthogonal', 'uniform', 'ones', 'normal'."
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_embedding:
            return x

        if self.embedding_only:
            return self.embedding_layer.weight

        return torch.cat([self.embedding_layer.weight, x], dim=-1)
