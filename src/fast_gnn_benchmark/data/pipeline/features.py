import numpy as np
import pandas as pd
import torch


def build_node_features(emb_pandas: pd.DataFrame, node2idx: dict[int, int]) -> torch.Tensor:
    """Construit la matrice de features des nœuds à partir des embeddings produits.

    Args:
        emb_pandas: Embeddings collectés sur le driver (colonnes internalId,
            embeddings), restreints aux produits présents dans le graphe.
        node2idx: Correspondance internalId -> index de nœud.

    Returns:
        Tensor x de forme [len(node2idx), embedding_dim]. La dimension est
        déduite du premier embedding. Un nœud sans embedding garde une ligne
        de zéros : il n'est pas retiré du graphe.
    """
    embedding_dim = len(emb_pandas["embeddings"].iloc[0])
    node_features = np.zeros((len(node2idx), embedding_dim), dtype=np.float32)

    positions = emb_pandas["internalId"].map(node2idx).values.astype(int)
    node_features[positions] = np.vstack(emb_pandas["embeddings"].values).astype(np.float32)

    return torch.tensor(node_features, dtype=torch.float)
