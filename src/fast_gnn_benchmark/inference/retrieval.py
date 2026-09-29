"""Top-k des candidats de la catégorie du trigger, le trigger lui-même exclu.

Toutes les stratégies renvoient la même forme : deux tableaux (n_triggers, k), node_id et score,
triés par score décroissant. Un rang sans candidat vaut -1 en node_id et NaN en score, et un trigger
non scoré (catégorie manquante ou pool vide) a sa ligne entière à -1.
"""

import math
from collections import defaultdict

import faiss
import numpy as np
import torch

from fast_gnn_benchmark.inference.triggers import Trigger

TopK = tuple[np.ndarray, np.ndarray]

# faiss avertit sous 39 vecteurs d'entraînement par centroïde
IVF_MIN_POINTS_PER_CENTROID = 39


def empty_topk(n_triggers: int, k: int) -> TopK:
    return np.full((n_triggers, k), -1, dtype=np.int64), np.full((n_triggers, k), np.nan, dtype=np.float32)


def exhaustive_topk(
    triggers: list[Trigger],
    candidates_by_category: dict[str, torch.Tensor],
    embeddings: torch.Tensor,
    classifier: torch.nn.Module,
    k: int,
    batch_size: int,
    device: str,
) -> TopK:
    """Score chaque paire (trigger, candidat) avec le classifieur, par lots de batch_size paires.

    Seule stratégie compatible avec hadamard_mlp : son score exige un forward par paire.

    Args:
        triggers: Triggers dont la catégorie est renseignée.
        candidates_by_category: Pools de candidats par catégorie.
        embeddings: Sortie du backbone pour tous les nœuds, sur device.
        classifier: Tête du modèle.
        k: Profondeur du classement.
        batch_size: Nombre de paires scorées par appel au classifieur.
        device: Device des embeddings.

    Returns:
        Tuple (top_ids, top_scores).
    """
    top_ids, top_scores = empty_topk(len(triggers), k)

    with torch.no_grad():
        for i, trigger in enumerate(triggers):
            if trigger["category"] is None:
                continue

            candidates = candidates_by_category[trigger["category"]]
            candidates = candidates[candidates != trigger["trigger_node_id"]]

            if candidates.numel() == 0:
                continue

            target_edges = torch.stack([torch.full_like(candidates, trigger["trigger_node_id"]), candidates])

            logits = torch.cat([
                classifier(embeddings, embeddings, target_edges[:, start:start + batch_size].to(device))
                for start in range(0, target_edges.shape[1], batch_size)
            ])

            top_logits, top_positions = torch.topk(logits, min(k, logits.numel()))
            n = top_logits.numel()
            top_ids[i, :n] = candidates[top_positions.cpu()].numpy()
            top_scores[i, :n] = top_logits.cpu().numpy()

    return top_ids, top_scores


def build_flat_index(vectors: np.ndarray) -> faiss.Index:
    """Index exact en produit scalaire : sur des vecteurs unitaires, c'est le score de la tête."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def ivf_nlist(pool_size: int) -> int:
    """Nombre de listes IVF : ~4 sqrt(n), borné pour garder 39 points par centroïde."""
    return max(1, min(pool_size // IVF_MIN_POINTS_PER_CENTROID, int(4 * math.sqrt(pool_size))))


def build_ivf_index(vectors: np.ndarray, nlist: int, nprobe: int, seed: int) -> faiss.Index:
    """Index approché IVF en produit scalaire, entraîné sur les vecteurs qu'il indexe."""
    dim = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.cp.seed = seed

    index.train(vectors)
    index.add(vectors)

    index.nprobe = nprobe
    return index


def build_indexes(
    vectors: np.ndarray,
    pools: dict[str, np.ndarray],
    ivf: bool,
    min_pool_for_ann: int = 2048,
    nprobe: int = 8,
    seed: int = 42,
) -> dict[str, faiss.Index]:
    """Construit un index par catégorie sur les vecteurs projetés de son pool.

    En IVF, un pool sous min_pool_for_ann reste en flat : l'exact y est déjà rapide et un IVF
    entraîné sur si peu de points perdrait du recall pour rien.

    Args:
        vectors: Vecteurs projetés de tous les nœuds, float32 contigu.
        pools: node_id candidats par catégorie.
        ivf: Construit des index IVF au lieu de flat.
        min_pool_for_ann: Taille de pool à partir de laquelle l'IVF est utilisé.
        nprobe: Nombre de listes visitées par requête IVF.
        seed: Graine du k-means IVF.

    Returns:
        Dictionnaire catégorie -> index.
    """
    indexes = {}
    for category, pool in pools.items():
        pool_vectors = np.ascontiguousarray(vectors[pool])
        if ivf and len(pool) >= min_pool_for_ann:
            indexes[category] = build_ivf_index(pool_vectors, ivf_nlist(len(pool)), nprobe, seed)
        else:
            indexes[category] = build_flat_index(pool_vectors)

    n_ivf = sum(1 for index in indexes.values() if isinstance(index, faiss.IndexIVF))
    print(f"index construits: {len(indexes)} dont {n_ivf} ivf")

    return indexes


def search(index: faiss.Index, pool: np.ndarray, vectors: np.ndarray, query_node_ids: np.ndarray, k: int) -> TopK:
    """Cherche k+1 voisins par requête puis retire le trigger lui-même, qui est dans son propre pool.

    Args:
        index: Index de la catégorie.
        pool: node_id indexés, dans l'ordre d'ajout à l'index.
        vectors: Vecteurs projetés de tous les nœuds.
        query_node_ids: node_id des triggers de la catégorie.
        k: Profondeur du classement.

    Returns:
        Tuple (top_ids, top_scores) de forme (len(query_node_ids), k).
    """
    scores, positions = index.search(np.ascontiguousarray(vectors[query_node_ids]), min(k + 1, index.ntotal))

    ids = np.where(positions >= 0, pool[np.clip(positions, 0, None)], -1)
    keep = (ids >= 0) & (ids != query_node_ids[:, None])

    # tri stable : les résultats gardés remontent en tête sans perdre l'ordre de score
    order = np.argsort(~keep, axis=1, kind="stable")
    ids_sorted = np.take_along_axis(ids, order, axis=1)
    scores_sorted = np.take_along_axis(scores, order, axis=1).astype(np.float32)

    invalid = np.arange(ids_sorted.shape[1])[None, :] >= keep.sum(axis=1)[:, None]
    ids_sorted = np.where(invalid, -1, ids_sorted)
    scores_sorted = np.where(invalid, np.nan, scores_sorted)

    top_ids, top_scores = empty_topk(len(query_node_ids), k)
    n = min(k, ids_sorted.shape[1])
    top_ids[:, :n] = ids_sorted[:, :n]
    top_scores[:, :n] = scores_sorted[:, :n]

    return top_ids, top_scores


def faiss_topk(
    triggers: list[Trigger], indexes: dict[str, faiss.Index], pools: dict[str, np.ndarray], vectors: np.ndarray, k: int
) -> TopK:
    """Top-k faiss, une recherche groupée par catégorie.

    Args:
        triggers: Triggers dont la catégorie est renseignée.
        indexes: Index par catégorie, issus de build_indexes.
        pools: node_id candidats par catégorie.
        vectors: Vecteurs projetés de tous les nœuds.
        k: Profondeur du classement.

    Returns:
        Tuple (top_ids, top_scores).
    """
    top_ids, top_scores = empty_topk(len(triggers), k)

    positions_by_category = defaultdict(list)
    for i, trigger in enumerate(triggers):
        if trigger["category"] in pools and len(pools[trigger["category"]]) > 0:
            positions_by_category[trigger["category"]].append(i)

    for category, positions in positions_by_category.items():
        query_node_ids = np.array([triggers[i]["trigger_node_id"] for i in positions], dtype=np.int64)
        ids, scores = search(indexes[category], pools[category], vectors, query_node_ids, k)
        top_ids[positions] = ids
        top_scores[positions] = scores

    return top_ids, top_scores
