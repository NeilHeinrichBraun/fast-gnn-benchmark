import json
from pathlib import Path
from typing import Any

import boto3
import torch
from pyspark.sql import DataFrame
from torch_geometric.data import Data

from fast_gnn_benchmark.schemas.pipeline_model import ArtifactParameters


def _staged(filename: str, parameters: ArtifactParameters) -> Path:
    """Prépare le chemin local de staging, en créant le répertoire si besoin."""
    path = Path(parameters.staging_path(filename))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _upload(path: Path, filename: str, parameters: ArtifactParameters) -> str:
    """Pousse un fichier de staging sur S3 et renvoie son URI."""
    boto3.client("s3").upload_file(str(path), parameters.bucket, parameters.key(filename))
    return parameters.uri(filename)


def _save_json(payload: Any, filename: str, parameters: ArtifactParameters) -> str:
    path = _staged(filename, parameters)
    path.write_text(json.dumps(payload))
    return _upload(path, filename, parameters)


def save_graph(data: Data, parameters: ArtifactParameters) -> str:
    """Sauvegarde l'objet PyG Data, en staging puis sur S3.

    Remplace CoViewDataset(InMemoryDataset) du prototype, qui n'était utilisé que pour appeler
    save(). CoViewMDMDataset relit indifféremment un Data nu ou la forme collatée de PyG.

    Args:
        data: Graphe assemblé par build_pyg_data.
        parameters: Destination des artefacts.

    Returns:
        L'URI S3 du fichier écrit.
    """
    path = _staged(parameters.graph_filename, parameters)
    torch.save(data, path)
    return _upload(path, parameters.graph_filename, parameters)


def save_node_mapping(node2idx: dict[int, int], parameters: ArtifactParameters) -> str:
    """Sauvegarde la correspondance internalId -> index de nœud.

    build_node_mapping renvoie des clés numpy (np.int64), non sérialisables en JSON : la
    conversion en int est obligatoire.

    Args:
        node2idx: Correspondance internalId -> index.
        parameters: Destination des artefacts.

    Returns:
        L'URI S3 du fichier écrit.
    """
    return _save_json(
        {int(node_id): int(idx) for node_id, idx in node2idx.items()},
        parameters.node_mapping_filename,
        parameters,
    )


def save_exec_mappings(exec_mappings: dict[str, Any], parameters: ArtifactParameters) -> str:
    """Sauvegarde les correspondances exec2code / code2exec des trois splits.

    Args:
        exec_mappings: Dictionnaire {split: {"exec2code": ..., "code2exec": ...}}.
        parameters: Destination des artefacts.

    Returns:
        L'URI S3 du fichier écrit.
    """
    return _save_json(exec_mappings, parameters.exec_mappings_filename, parameters)


def save_manifest(manifest: dict[str, Any], parameters: ArtifactParameters) -> str:
    """Sauvegarde le manifest du run : config, comptes par split, num_nodes, MRR de la prod.

    Args:
        manifest: Contenu du manifest.
        parameters: Destination des artefacts.

    Returns:
        L'URI S3 du fichier écrit.
    """
    return _save_json(manifest, parameters.manifest_filename, parameters)


def write_parquet(df: DataFrame, filename: str, parameters: ArtifactParameters) -> str:
    """Écrit un DataFrame Spark directement sur S3, en écrasant l'existant.

    Args:
        df: Table à écrire.
        filename: Nom du parquet, issu de ArtifactParameters.
        parameters: Destination des artefacts.

    Returns:
        L'URI S3 du parquet écrit.
    """
    uri = parameters.uri(filename)
    df.write.mode("overwrite").parquet(uri)
    return uri
