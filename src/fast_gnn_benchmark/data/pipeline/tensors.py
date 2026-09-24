import numpy as np
import pandas as pd
import torch


def edges_to_tensors(df: pd.DataFrame, symmetric: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Convertit un DataFrame d'arêtes remappées en tensors compatibles PyG.

    Args:
        df: DataFrame d'arêtes remappées (colonnes product_A, product_B,
            weight), avec des index de nœuds contigus.
        symmetric: Si True, ajoute l'arête inverse (B->A) pour chaque
            arête (A->B), pour un graphe non dirigé.

    Returns:
        Tuple (edge_index, edge_attr). edge_index a la forme [2, E] (ou
        [2, 2E] si symmetric), edge_attr la forme [E] (ou [2E]) avec les
        poids dupliqués à l'identique dans les deux sens.
    """
    src = df["product_A"].values
    dst = df["product_B"].values
    weight = df["weight"].values

    if symmetric:
        edge_index = np.stack([
            np.concatenate([src, dst]),
            np.concatenate([dst, src]),
        ])
        edge_attr = np.concatenate([weight, weight])
    else:
        edge_index = np.stack([src, dst])
        edge_attr = weight

    return (
        torch.tensor(edge_index, dtype=torch.long),
        torch.tensor(edge_attr, dtype=torch.float),
    )
