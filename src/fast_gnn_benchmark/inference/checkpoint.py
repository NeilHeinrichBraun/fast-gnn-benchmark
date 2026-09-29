import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from fast_gnn_benchmark.models.link_prediction import LinkPredictionModel
from fast_gnn_benchmark.schemas.inference_model import CheckpointParameters


def list_checkpoints(ckpt_dir: str) -> list[tuple[float, str]]:
    """Liste les checkpoints de ckpt_dir, du plus récent au plus ancien, et les affiche.

    Args:
        ckpt_dir: Dossier de checkpoints (.ckpt) à lister.

    Returns:
        Liste de tuples (mtime, nom_fichier), triée par mtime décroissant.
    """
    files = sorted(
        (
            (path.stat().st_mtime, path.name)
            for path in Path(ckpt_dir).iterdir()
            if path.suffix == ".ckpt"
        ),
        reverse=True,
    )

    for mtime, fname in files:
        print(f"{datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M:%S}  {fname}")

    return files


def resolve_checkpoint_path(parameters: CheckpointParameters) -> str:
    """Résout "latest" vers le .ckpt le plus récent de parameters.dir, ou renvoie le chemin fourni.

    save_top_k=1 sur val/mrr_trigger : le dossier ne contient que le checkpoint du meilleur epoch,
    dont le nom dépend de l'epoch atteint.

    Args:
        parameters: Dossier et checkpoint demandé.

    Returns:
        Chemin du checkpoint retenu.

    Raises:
        FileNotFoundError: Si "latest" est demandé et que le dossier ne contient aucun .ckpt.
    """
    if parameters.path != "latest":
        return parameters.path

    files = list_checkpoints(parameters.dir)
    if not files:
        raise FileNotFoundError(f"Aucun .ckpt dans {parameters.dir}")

    return os.path.join(parameters.dir, files[0][1])


def load_checkpoint(
    parameters: CheckpointParameters, device: str, require_projection: bool
) -> tuple[LinkPredictionModel, dict[str, Any]]:
    """Charge le modèle en mode eval et collecte ses infos de diagnostic pour le manifest.

    Contrat des têtes indexables : project() renvoie un vecteur unitaire par nœud dont le produit
    scalaire est le score. hadamard_mlp ne l'a pas, son score n'étant pas décomposable.

    Args:
        parameters: Checkpoint à charger.
        device: Device torch sur lequel charger le modèle.
        require_projection: Exige une tête exposant project(), pour les stratégies faiss.

    Returns:
        Tuple (model, info) : le modèle chargé et un dictionnaire sérialisable en JSON (chemin,
        epoch, global_step, best_model_score, tête, nombre de paramètres).

    Raises:
        ValueError: Si require_projection et que la tête n'expose pas project().
    """
    checkpoint_path = resolve_checkpoint_path(parameters)
    print(f"checkpoint: {checkpoint_path}")

    raw_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = LinkPredictionModel.load_from_checkpoint(checkpoint_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)

    head = model.model.classifier
    if require_projection and not hasattr(head, "project"):
        raise ValueError(
            f"Tête {type(head).__name__} non indexable : les stratégies faiss exigent cosine_similarity "
            f"ou mlp_cosine (checkpoint: {checkpoint_path})"
        )

    best_model_score = next(
        (state["best_model_score"] for state in raw_ckpt["callbacks"].values() if "best_model_score" in state),
        None,
    )

    info = {
        "path": checkpoint_path,
        "epoch": raw_ckpt["epoch"],
        "global_step": raw_ckpt["global_step"],
        "best_model_score": None if best_model_score is None else float(best_model_score),
        "head": type(head).__name__,
        "num_params": sum(p.numel() for p in model.parameters()),
    }

    for key, value in info.items():
        print(f"{key}: {value}")
    print(model.hparams.model_parameters)

    return model, info
