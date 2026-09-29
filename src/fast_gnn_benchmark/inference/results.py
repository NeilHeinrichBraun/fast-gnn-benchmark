import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, IntegerType, LongType, StructField, StructType

from fast_gnn_benchmark.inference.triggers import Trigger

BATCH_TRIGGERS = 20_000

MODEL_ROWS_SCHEMA = StructType([
    StructField("exec_code", LongType(), True),
    StructField("internal_id", LongType(), True),
    StructField("score", DoubleType(), True),
    StructField("rank", IntegerType(), True),
])

KEYS_SCHEMA = StructType([
    StructField("exec_code", LongType(), True),
    StructField("trigger_internal_id", LongType(), True),
])

EMPTY_PRODUCTS_TYPE = "array<struct<internal_id:bigint,score:double,rank:int>>"


def stage_model_rows(
    spark: SparkSession,
    triggers: list[Trigger],
    top_ids: np.ndarray,
    top_scores: np.ndarray,
    idx2node: dict[int, int],
    path: str,
    batch_triggers: int = BATCH_TRIGGERS,
) -> None:
    """Écrit une ligne plate par (trigger, rang) en parquet, par lots de triggers.

    Un createDataFrame sur tous les triggers d'un coup sérialise toute la liste sur le driver.
    Seul le premier lot écrase, les suivants ajoutent.

    Args:
        spark: Session Spark active.
        triggers: Triggers scorés, alignés sur les lignes de top_ids.
        top_ids: node_id du top-k, -1 pour un rang vide.
        top_scores: Scores du top-k.
        idx2node: Correspondance index de nœud -> internalId.
        path: URI du parquet de staging.
        batch_triggers: Nombre de triggers par lot.
    """
    mode = "overwrite"

    for start in range(0, len(triggers), batch_triggers):
        end = min(start + batch_triggers, len(triggers))
        rows = [
            (triggers[i]["exec_code"], int(idx2node[int(node_id)]), float(score), rank)
            for i in range(start, end)
            for rank, (node_id, score) in enumerate(zip(top_ids[i], top_scores[i], strict=True), start=1)
            if node_id >= 0
        ]
        if not rows:
            continue

        spark.createDataFrame(rows, schema=MODEL_ROWS_SCHEMA, verifySchema=False).write.mode(mode).parquet(path)
        print(f"  lot {start}-{end}: {len(rows)} lignes")
        mode = "append"

    if mode == "overwrite":
        spark.createDataFrame([], schema=MODEL_ROWS_SCHEMA).write.mode("overwrite").parquet(path)
        print("  aucun trigger score, parquet vide ecrit")


def build_model_results(
    spark: SparkSession, triggers: list[Trigger], rows_path: str, sessions_raw: DataFrame
) -> DataFrame:
    """Assemble model_results au format de prod_results : un classement par trigger, découpé en
    positifs (produits vus dans la session du trigger) et négatifs.

    Un trigger non scoré reste présent avec des listes vides : la population est celle de la table
    prod, ce qui garde les MRR coverage-adjusted comparables.

    Args:
        spark: Session Spark active.
        triggers: Tous les triggers du graphe, scorés ou non.
        rows_path: Parquet écrit par stage_model_rows.
        sessions_raw: Table sessions_raw du split, pour les produits vus par session.

    Returns:
        DataFrame exec_code, trigger_internal_id, positives, negatives, products_returned.
    """
    df_keys = spark.createDataFrame(
        [(t["exec_code"], t["trigger_internal_id"]) for t in triggers], schema=KEYS_SCHEMA, verifySchema=False
    )

    df_products = (
        spark.read.parquet(rows_path)
        .groupBy("exec_code")
        .agg(F.sort_array(F.collect_list(F.struct("rank", "internal_id", "score"))).alias("ranked"))
        .withColumn(
            "products_returned",
            F.transform(
                "ranked",
                lambda r: F.struct(
                    r["internal_id"].alias("internal_id"),
                    r["score"].alias("score"),
                    r["rank"].alias("rank"),
                ),
            ),
        )
        .select("exec_code", "products_returned")
    )

    session_ids_by_exec_code = (
        sessions_raw
        .select("exec_code", F.col("session_products.internal_id").alias("session_ids"))
        .dropDuplicates(["exec_code"])
    )

    return (
        df_keys
        .join(df_products, on="exec_code", how="left")
        .join(session_ids_by_exec_code, on="exec_code", how="left")
        .withColumn("products_returned", F.coalesce(F.col("products_returned"), F.array().cast(EMPTY_PRODUCTS_TYPE)))
        .withColumn("session_ids", F.coalesce(F.col("session_ids"), F.array().cast("array<bigint>")))
        .withColumn(
            "positives",
            F.filter("products_returned", lambda p: F.array_contains(F.col("session_ids"), p["internal_id"])),
        )
        .withColumn(
            "negatives",
            F.filter("products_returned", lambda p: ~F.array_contains(F.col("session_ids"), p["internal_id"])),
        )
        .select("exec_code", "trigger_internal_id", "positives", "negatives", "products_returned")
    )
