from typing import Any

import numpy as np
import torch
import yaml
from pyspark.sql import DataFrame, SparkSession

from fast_gnn_benchmark.data.dataset.coview_mdm import CoViewMDMDataset
from fast_gnn_benchmark.data.pipeline import artifacts, metrics
from fast_gnn_benchmark.inference import checkpoint, embeddings, results, retrieval, triggers, validation
from fast_gnn_benchmark.schemas.inference_model import FAISS_STRATEGIES, InferenceParameters


def get_inference_parameters_from_config(config_file: str) -> InferenceParameters:
    """Charge et valide une configuration d'inférence.

    Args:
        config_file: Chemin du YAML de configuration.

    Returns:
        La configuration validée.
    """
    with open(config_file) as file:
        return InferenceParameters.model_validate(yaml.safe_load(file))


def run_faiss(
    parameters: InferenceParameters,
    trigger_list: list[triggers.Trigger],
    candidates_by_category: dict[str, torch.Tensor],
    projected: torch.Tensor,
) -> tuple[retrieval.TopK, dict[str, Any]]:
    """Retrieval faiss, contrôlé contre une référence torch et, en IVF, contre l'index exact.

    L'index exact est toujours construit : il porte le contrôle de parité torch, et l'IVF n'a de
    recall mesurable que contre lui.

    Args:
        parameters: Configuration d'inférence.
        trigger_list: Triggers dont la catégorie est renseignée.
        candidates_by_category: Pools de candidats par catégorie.
        projected: Vecteurs projetés de tous les nœuds, sortie de head.project().

    Returns:
        Tuple (topk, checks) : le top-k de la stratégie demandée et les contrôles pour le manifest.
    """
    rp = parameters.retrieval
    vectors = np.ascontiguousarray(projected.detach().cpu().numpy(), dtype="float32")
    pools = {category: pool.numpy().astype(np.int64) for category, pool in candidates_by_category.items()}

    flat = retrieval.faiss_topk(
        trigger_list, retrieval.build_indexes(vectors, pools, ivf=False), pools, vectors, parameters.top_k
    )
    checks: dict[str, Any] = {
        "reference": validation.check_faiss_against_reference(
            trigger_list, candidates_by_category, projected, *flat, rp.reference_sample_size, rp.reference_atol
        )
    }

    if rp.strategy == "faiss_flat":
        return flat, checks

    indexes = retrieval.build_indexes(
        vectors, pools, ivf=True, min_pool_for_ann=rp.min_pool_for_ann, nprobe=rp.nprobe, seed=rp.ivf_seed
    )
    ivf = retrieval.faiss_topk(trigger_list, indexes, pools, vectors, parameters.top_k)
    checks["recall_vs_flat"] = validation.recall_against_exact(ivf[0], flat[0], rp.min_recall_vs_flat)

    return ivf, checks


def do_inference(spark: SparkSession, parameters: InferenceParameters) -> DataFrame:
    """Score tous les triggers d'un split et écrit model_results et son manifest sur S3.

    Args:
        spark: Session Spark active.
        parameters: Configuration d'inférence.

    Returns:
        La table model_results écrite sur S3.
    """
    art = parameters.artifacts
    split = parameters.split
    name = parameters.output_name
    use_faiss = parameters.retrieval.strategy in FAISS_STRATEGIES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | stratégie: {parameters.retrieval.strategy} | split: {split}")

    model, checkpoint_info = checkpoint.load_checkpoint(parameters.checkpoint, device, require_projection=use_faiss)

    dataset = CoViewMDMDataset(bucket=art.bucket, s3_key=art.key(art.graph_filename))
    node2idx = artifacts.load_node_mapping(art)
    idx2node = {idx: node_id for node_id, idx in node2idx.items()}
    print(f"num_nodes: {dataset.num_nodes} | node2idx entries: {len(node2idx)}")

    prod_results = spark.read.parquet(art.uri(art.prod_results_filename(split)))
    sessions_raw = spark.read.parquet(art.uri(art.sessions_raw_filename(split)))

    node_embeddings = embeddings.compute_node_embeddings(dataset, model, device)
    print(f"x shape: {tuple(node_embeddings.shape)}")

    trigger_list, skipped_no_node = triggers.extract_all_triggers(prod_results, node2idx)
    missing_category = triggers.extract_categories(spark, trigger_list, parameters.customer_shortname)
    candidates_by_category = triggers.build_candidates_by_category(
        spark, trigger_list, node2idx, parameters.customer_shortname
    )

    checks: dict[str, Any] = {}
    if use_faiss:
        with torch.no_grad():
            projected = model.model.classifier.project(node_embeddings)
        del node_embeddings
        (top_ids, top_scores), checks = run_faiss(parameters, trigger_list, candidates_by_category, projected)
    else:
        top_ids, top_scores = retrieval.exhaustive_topk(
            trigger_list,
            candidates_by_category,
            node_embeddings,
            model.model.classifier,
            parameters.top_k,
            parameters.retrieval.batch_size,
            device,
        )

    n_unscored = int((top_ids[:, 0] == -1).sum())
    print(f"triggers non scores (categorie manquante ou pool vide): {n_unscored}/{len(trigger_list)}")

    rows_path = art.uri(art.model_rows_staging_filename(split, name))
    results.stage_model_rows(spark, trigger_list, top_ids, top_scores, idx2node, rows_path)
    model_results = results.build_model_results(spark, trigger_list, rows_path, sessions_raw)

    output_uri = art.uri(art.model_results_filename(split, name))
    model_results.write.mode("overwrite").parquet(output_uri)
    print(f"{output_uri} uploaded")

    # relu depuis S3 : mesure ce qui a été écrit, sans recalculer la chaîne de jointures
    model_results = spark.read.parquet(output_uri)
    model_results.printSchema()

    manifest = {
        "parameters": parameters.model_dump(mode="json"),
        "checkpoint": checkpoint_info,
        "num_nodes": dataset.num_nodes,
        "triggers": {
            "prod": skipped_no_node + len(trigger_list),
            "outside_graph": skipped_no_node,
            "missing_category": missing_category,
            "in_graph": len(trigger_list),
            "unscored": n_unscored,
        },
        "checks": checks,
        "model_mrr": metrics.production_mrr_prototype(model_results, "positives"),
        "output": output_uri,
    }

    manifest_uri = artifacts.save_json(manifest, art.inference_manifest_filename(split, name), art)
    print(f"{manifest_uri} uploaded")

    return model_results
