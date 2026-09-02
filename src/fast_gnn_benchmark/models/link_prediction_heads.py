import torch
import torch.nn.functional as F


class CosineSimilarityClassifier(torch.nn.Module):
    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Vecteurs normalises par noeud: ce sont eux qu'on indexe en ANN."""
        return F.normalize(embeddings, dim=-1)

    def forward(
        self, embedding_1: torch.Tensor, embedding_2: torch.Tensor, edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        e1 = F.normalize(embedding_1[edge_label_index[0]], dim=-1)
        e2 = F.normalize(embedding_2[edge_label_index[1]], dim=-1)
        return (e1 * e2).sum(dim=-1)


class MLP_CosinePredictor(torch.nn.Module):
    """Tour MLP partagee par les deux cotes de la paire, puis cosine.

    Le score reste un produit scalaire entre deux vecteurs par noeud, donc indexable en ANN
    via `project`, contrairement a `Hadamard_MLPPredictor` qui exige un forward par paire.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
        num_layers: int = 2,
        use_residual: bool = False,
        use_layer_norm: bool = False,
        normalize_embeddings: bool = False,
    ):
        super().__init__()

        self.linear_layers = torch.nn.ModuleList(
            torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        )

        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.normalize_embeddings = normalize_embeddings

        if use_layer_norm:
            self.layer_norms = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.layer_norms.append(torch.nn.LayerNorm(hidden_dim))

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Vecteurs normalises par noeud: ce sont eux qu'on indexe en ANN."""
        x = embeddings
        if self.normalize_embeddings:
            x = F.normalize(x, dim=-1)

        ori = x
        for i in range(len(self.linear_layers) - 1):
            x = self.linear_layers[i](x)
            if self.use_residual:
                x = x + ori
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # pas d'activation apres la derniere couche
        return F.normalize(self.linear_layers[-1](x), dim=-1)

    def forward(
        self, embedding_1: torch.Tensor, embedding_2: torch.Tensor, edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        # on projette les tenseurs de noeuds entiers avant d'indexer les paires: il y a bien moins
        # de noeuds que d'aretes cibles, et le gather accumule les gradients des deux roles
        h1 = self.project(embedding_1)
        h2 = h1 if embedding_2 is embedding_1 else self.project(embedding_2)

        return (h1[edge_label_index[0]] * h2[edge_label_index[1]]).sum(dim=-1)


class Hadamard_MLPPredictor(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        num_layers: int = 2,
        use_residual: bool = False,
        use_layer_norm: bool = False,
        normalize_embeddings: bool = False,
    ):
        super().__init__()

        self.linear_layers = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.linear_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.linear_layers.append(torch.nn.Linear(hidden_dim, 1))

        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.normalize_embeddings = normalize_embeddings

        if use_layer_norm:
            self.layer_norms = torch.nn.ModuleList()
            for _ in range(num_layers - 1):
                self.layer_norms.append(torch.nn.LayerNorm(hidden_dim))

    def forward(self, embedding_1: torch.Tensor, embedding_2: torch.Tensor, edge_label_index: torch.Tensor):

        e1 = embedding_1[edge_label_index[0]]
        e2 = embedding_2[edge_label_index[1]]

        if self.normalize_embeddings:
            e1 = F.normalize(e1, dim=-1)
            e2 = F.normalize(e2, dim=-1)
        x = e1 * e2

        ori = x
        for i in range(len(self.linear_layers) - 1):
            x = self.linear_layers[i](x)
            if self.use_residual:
                x += ori
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.linear_layers[-1](x).squeeze(-1)
