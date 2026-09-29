import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor

from fast_gnn_benchmark.data.dataset.coview_mdm import CoViewMDMDataset
from fast_gnn_benchmark.models.link_prediction import LinkPredictionModel


def reset_conv_caches(backbone: torch.nn.Module) -> None:
    """Vide les caches de normalisation des convolutions (cached: true dans les configs GCN).

    Un cache rempli à l'entraînement sur un autre edge_index serait réutilisé tel quel au forward.
    """
    for module in backbone.modules():
        if hasattr(module, "_cached_edge_index"):
            module._cached_edge_index = None
        if hasattr(module, "_cached_adj_t"):
            module._cached_adj_t = None


def compute_node_embeddings(dataset: CoViewMDMDataset, model: LinkPredictionModel, device: str) -> torch.Tensor:
    """Forward pass GNN complet (embedder + backbone), calculé une seule fois pour tous les triggers.

    Args:
        dataset: Dataset dont data.x et data.edge_index alimentent le forward.
        model: Modèle chargé ; son embedder et son backbone sont utilisés.
        device: Device torch sur lequel calculer les embeddings.

    Returns:
        Embeddings de tous les nœuds (un par node_id de node2idx), sortie du backbone : pas encore
        projetés dans l'espace du classifieur.
    """
    adj_t = ToSparseTensor()(Data(edge_index=dataset.data.edge_index, num_nodes=dataset.num_nodes)).adj_t.to(device)

    reset_conv_caches(model.model.backbone)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()

    with torch.no_grad():
        x = model.model.embedder(dataset.data.x.to(device))
        return model.model.backbone(x, adj_t)
