from typing import Any

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from fast_gnn_benchmark.data.pipeline import artifacts
from fast_gnn_benchmark.data.pipeline.runner import resolve_customer
from fast_gnn_benchmark.evaluation import aggregates, assembly, validation
from fast_gnn_benchmark.evaluation.assembly import PROD_ARMS
from fast_gnn_benchmark.schemas.inference_model import InferenceParameters


def do_stats(spark: SparkSession, parameters: InferenceParameters) -> dict[str, Any]:
    """Compare le modèle d'un run d'inférence aux deux variantes prod et écrit les résultats sur S3.

    Prend la config du job d'inférence : output_name, split et artifacts désignent exactement le
    model_results à évaluer et les tables prod qui ont servi à le construire.

    Écrit trois artefacts : product_metadata (nom et image des produits cités), trigger_metrics (une
    ligne par trigger avec les métriques des trois bras, pour l'analyse fine en notebook) et stats
    (tous les agrégats, en JSON).

    Args:
        spark: Session Spark active.
        parameters: Configuration du job d'inférence évalué.

    Returns:
        Le contenu du JSON de statistiques.
    """
    art, split, name = parameters.artifacts, parameters.split, parameters.output_name

    sessions_raw = spark.read.parquet(art.uri(art.sessions_raw_filename(split)))
    prod_results = spark.read.parquet(art.uri(art.prod_results_filename(split)))
    model_results = spark.read.parquet(art.uri(art.model_results_filename(split, name)))

    print(f"sessions_raw: {sessions_raw.count()} triggers")
    print(f"prod_results: {prod_results.count()} triggers")
    print(f"model_results: {model_results.count()} triggers")

    compared = validation.check_alignment(prod_results, model_results)

    customer = resolve_customer(spark, parameters.customer_shortname)
    product_metadata = assembly.build_product_metadata(
        spark, sessions_raw, prod_results, model_results, customer.db_names
    ).cache()
    n_products = product_metadata.count()
    n_unresolved = product_metadata.filter(F.col("name").isNull()).count()
    print(f"produits distincts: {n_products} | non resolus dans le catalogue: {n_unresolved}")

    product_metadata_uri = art.uri(art.product_metadata_filename(split, name))
    product_metadata.write.mode("overwrite").parquet(product_metadata_uri)
    print(f"{product_metadata_uri} uploaded")
    product_metadata.unpersist()

    trigger_metrics_uri = art.uri(art.trigger_metrics_filename(split, name))
    assembly.add_metrics(assembly.assemble_arms(prod_results, model_results, sessions_raw)).write.mode(
        "overwrite"
    ).parquet(trigger_metrics_uri)
    print(f"{trigger_metrics_uri} uploaded")

    # relu depuis S3 : toutes les agrégations partent de la table écrite, sans recalculer les jointures
    trigger_metrics = spark.read.parquet(trigger_metrics_uri).cache()

    population = trigger_metrics.agg(
        F.count("*").alias("triggers"),
        F.count(F.when(F.col("missing_model"), 1)).alias("missing_model"),
        F.count(F.when(~F.col("model_answered"), 1)).alias("model_unanswered"),
        F.count(F.when(F.col("session_size").isNull() | (F.col("session_size") <= 0), 1)).alias("empty_session"),
    ).collect()[0].asDict()
    print(f"population: {population}")

    stats = {
        "parameters": parameters.model_dump(mode="json"),
        "inputs": {
            "model_results": art.uri(art.model_results_filename(split, name)),
            "inference_manifest": art.uri(art.inference_manifest_filename(split, name)),
        },
        "alignment_compared": compared,
        "products": {"distinct": n_products, "unresolved": n_unresolved},
        "population": population,
        "metrics": aggregates.aggregate_metrics(trigger_metrics),
        "matrices": {arm: aggregates.matrix_2x2(trigger_metrics, arm) for arm in PROD_ARMS},
        "overlap": aggregates.product_overlap(trigger_metrics),
        "by_prod_session": {arm: aggregates.by_prod_session(trigger_metrics, arm) for arm in PROD_ARMS},
        "variant_crossing": aggregates.variant_crossing(trigger_metrics),
        "matrices_by_session": aggregates.matrices_by_session(trigger_metrics),
        "session_level": aggregates.session_level(sessions_raw, trigger_metrics),
        "outputs": {"product_metadata": product_metadata_uri, "trigger_metrics": trigger_metrics_uri},
    }

    stats_uri = artifacts.save_json(stats, art.stats_filename(split, name), art)
    print(f"{stats_uri} uploaded")

    return stats
