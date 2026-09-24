import pandas as pd
import torch

EDGE_TENSOR_ROWS = 3
EDGE_TENSOR_DIMS = 2


def check_remapped_edges(df: pd.DataFrame, name: str) -> None:
    """Vérifie qu'aucune arête n'a perdu un produit lors du remap.

    remap_edges laisse un NaN pour tout internalId absent de node2idx, que le dropna en aval
    fait disparaître sans le compter.

    Args:
        df: Arêtes remappées (colonnes product_A, product_B).
        name: Nom du jeu d'arêtes, pour le message d'erreur.

    Raises:
        ValueError: Si au moins une arête contient un produit non mappé.
    """
    missing = int(df[["product_A", "product_B"]].isna().any(axis=1).sum())
    if missing:
        raise ValueError(f"{name} : {missing} arêtes dont un produit est absent de node2idx")


def check_edge_tensor(tensor: torch.Tensor, name: str, num_nodes: int) -> None:
    """Vérifie le format [3, N] lu par TriggerLoader et la validité des index de nœuds.

    TriggerLoader indexe positionnellement (ligne 0 = exec_code, lignes 1-2 = trigger et
    candidat) : une forme inattendue ne lève rien à l'entraînement, elle donne des résultats
    faux.

    Args:
        tensor: Tensor de paires labellisées.
        name: Nom du tensor, pour le message d'erreur.
        num_nodes: Nombre de nœuds du graphe.

    Raises:
        ValueError: Si la forme n'est pas [3, N] ou si un index de nœud sort de [0, num_nodes).
    """
    if tensor.ndim != EDGE_TENSOR_DIMS or tensor.shape[0] != EDGE_TENSOR_ROWS:
        raise ValueError(f"{name} : forme {tuple(tensor.shape)} au lieu de [{EDGE_TENSOR_ROWS}, N]")

    nodes = tensor[1:]
    if nodes.numel() and (int(nodes.min()) < 0 or int(nodes.max()) >= num_nodes):
        raise ValueError(f"{name} : index de nœud hors de [0, {num_nodes})")


def check_node_features(x: torch.Tensor, max_missing_ratio: float = 1.0) -> int:
    """Compte les nœuds sans embedding et refuse au-delà du seuil.

    Args:
        x: Matrice de features des nœuds.
        max_missing_ratio: Part maximale tolérée de lignes entièrement nulles.

    Returns:
        Le nombre de nœuds sans embedding.

    Raises:
        ValueError: Si la part dépasse max_missing_ratio.
    """
    missing = int((x == 0).all(dim=1).sum())
    ratio = missing / x.shape[0]

    if ratio > max_missing_ratio:
        raise ValueError(f"{ratio:.1%} de nœuds sans embedding, seuil {max_missing_ratio:.1%}")

    return missing


def check_counts_match(actual: int, expected: int, label: str) -> None:
    """Vérifie qu'une jointure n'a rien perdu.

    Args:
        actual: Nombre de lignes après jointure.
        expected: Nombre de lignes attendu.
        label: Nom de la table, pour le message d'erreur.

    Raises:
        ValueError: Si les deux comptes diffèrent.
    """
    if actual != expected:
        raise ValueError(f"{label} : {expected - actual} lignes perdues ({actual} au lieu de {expected})")


def check_num_nodes(num_nodes: int, expected: int | None) -> None:
    """Confronte le nombre de nœuds à la valeur attendue en configuration.

    num_nodes est recopié à la main dans les configs d'entraînement : un écart silencieux y
    produit un embedder de mauvaise taille.

    Args:
        num_nodes: Nombre de nœuds effectivement construit.
        expected: Valeur attendue, ou None pour ne rien vérifier.

    Raises:
        ValueError: Si les deux valeurs diffèrent.
    """
    if expected is not None and num_nodes != expected:
        raise ValueError(f"num_nodes = {num_nodes}, attendu {expected}")
