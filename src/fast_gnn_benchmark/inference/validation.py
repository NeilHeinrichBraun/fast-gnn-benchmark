import numpy as np
import torch

from fast_gnn_benchmark.inference.triggers import Trigger


def check_faiss_against_reference(
    triggers: list[Trigger],
    candidates_by_category: dict[str, torch.Tensor],
    projected: torch.Tensor,
    top_ids: np.ndarray,
    top_scores: np.ndarray,
    sample_size: int,
    atol: float,
) -> dict[str, int]:
    """Confronte le top-k faiss exact à un produit scalaire torch sur un échantillon de triggers.

    Le produit scalaire est calculé sur les vecteurs déjà projetés : c'est exactement ce que calcule
    faiss. Repasser par classifier() réappliquerait project(), invisible pour la cosine nue
    (renormaliser est idempotent) mais faux pour mlp_cosine.

    La comparaison porte sur les scores triés et non sur les node_id : deux produits d'embeddings
    identiques sont à égalité, et torch et faiss ne départagent pas les ex aequo de la même façon.

    Args:
        triggers: Triggers scorés.
        candidates_by_category: Pools de candidats par catégorie.
        projected: Vecteurs projetés de tous les nœuds, sortie de head.project().
        top_ids: node_id faiss, (n_triggers, k).
        top_scores: Scores faiss, (n_triggers, k).
        sample_size: Nombre de triggers vérifiés, pris dans l'ordre.
        atol: Tolérance absolue sur les scores.

    Returns:
        Dictionnaire checked, same_ids (triggers au même ensemble de node_id).

    Raises:
        ValueError: Si les scores d'au moins un trigger divergent.
    """
    k = top_ids.shape[1]
    positions = [i for i, t in enumerate(triggers) if t["category"] is not None][:sample_size]

    checked, same_ids, mismatches = 0, 0, []

    with torch.no_grad():
        for i in positions:
            trigger = triggers[i]
            candidates = candidates_by_category[trigger["category"]]
            candidates = candidates[candidates != trigger["trigger_node_id"]]
            if candidates.numel() == 0:
                continue

            logits = projected[candidates.to(projected.device)] @ projected[trigger["trigger_node_id"]]
            ref_scores, ref_positions = torch.topk(logits, min(k, logits.numel()))
            ref_ids = candidates[ref_positions.cpu()].numpy()
            ref_scores = ref_scores.cpu().numpy()

            valid = top_ids[i] >= 0
            checked += 1
            same_ids += set(ref_ids.tolist()) == set(top_ids[i][valid].tolist())

            if valid.sum() != len(ref_scores) or not np.allclose(ref_scores, top_scores[i][valid], atol=atol):
                mismatches.append(trigger["exec_code"])

    print(f"référence torch: {checked - len(mismatches)}/{checked} scores identiques, {same_ids}/{checked} mêmes node_id")

    if mismatches:
        raise ValueError(f"{len(mismatches)}/{checked} triggers divergent de la référence torch, ex. {mismatches[:5]}")

    return {"checked": checked, "same_ids": same_ids}


def recall_against_exact(top_ids: np.ndarray, top_ids_exact: np.ndarray, min_recall: float | None) -> float:
    """recall@k moyen d'un top-k approché contre le top-k exact, par trigger.

    Un trigger sans réponse exacte compte pour 1 : il n'y avait rien à retrouver.

    Args:
        top_ids: node_id approchés, (n_triggers, k).
        top_ids_exact: node_id exacts, (n_triggers, k).
        min_recall: Seuil sous lequel lever, ou None pour seulement mesurer.

    Returns:
        Le recall moyen.

    Raises:
        ValueError: Si le recall est sous min_recall.
    """
    valid = top_ids_exact >= 0
    hits = ((top_ids[:, :, None] == top_ids_exact[:, None, :]) & valid[:, None, :]).any(axis=1).sum(axis=1)
    per_trigger = np.where(valid.any(axis=1), hits / np.maximum(valid.sum(axis=1), 1), 1.0)
    recall = float(per_trigger.mean()) if len(per_trigger) else 1.0

    print(f"recall@{top_ids.shape[1]} moyen (ivf vs exact): {recall:.4f}")
    print(f"triggers a recall < 1.0: {(per_trigger < 1.0).sum()}/{len(per_trigger)}")

    if min_recall is not None and recall < min_recall:
        raise ValueError(f"recall ivf {recall:.4f} sous le seuil {min_recall}")

    return recall
