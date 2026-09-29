"""Assemblage des trois bras comparés sur la même population de triggers.

| bras      | origine                                                                    |
|-----------|----------------------------------------------------------------------------|
| model     | top-k du modèle, classé sur tout le pool de la catégorie du trigger         |
| display   | 12 premiers de relevantProducts dans l'ordre d'affichage                    |
| relevance | 12 premiers de relevantProducts par relevanceScore décroissant              |
"""

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from fast_gnn_benchmark.data.pipeline.sources import PRODUCT_TABLE

ARMS = ("model", "display", "relevance")
PROD_ARMS = ("display", "relevance")
EMPTY_MODEL_TYPE = "array<struct<internal_id:bigint,score:double,rank:int>>"


def _product_ids(df: DataFrame, column: str) -> DataFrame:
    return df.select(F.explode(column).alias("product")).select(F.col("product.internal_id").alias("internal_id"))


def build_product_metadata(
    spark: SparkSession,
    sessions_raw: DataFrame,
    prod_results: DataFrame,
    model_results: DataFrame,
    db_names: list[str],
) -> DataFrame:
    """Nom et image de tout produit apparaissant dans une session, un bras ou comme trigger.

    Le catalogue n'est pas filtré sur isActive : un produit désactivé depuis le run reste
    affichable dans les analyses.

    Args:
        spark: Session Spark active.
        sessions_raw: Table sessions_raw du split.
        prod_results: Table prod du split.
        model_results: Table du modèle, sortie du job d'inférence.
        db_names: Bases clients du catalogue.

    Returns:
        DataFrame internal_id, name, imageUrl, une ligne par produit ; name et imageUrl sont null
        pour un produit absent du catalogue.
    """
    product_ids = (
        _product_ids(sessions_raw, "session_products")
        .union(_product_ids(prod_results, "products_returned_display"))
        .union(_product_ids(prod_results, "products_returned_relevance"))
        .union(_product_ids(model_results, "products_returned"))
        .union(sessions_raw.select(F.col("trigger_internal_id").alias("internal_id")))
        .distinct()
    )

    df_products = (
        spark.table(PRODUCT_TABLE)
        .filter(F.col("db_name").isin(db_names))
        .select(F.col("internalId").cast("bigint").alias("internal_id"), "name", "imageUrl")
        .dropDuplicates(["internal_id"])
    )

    return product_ids.join(df_products, on="internal_id", how="left")


def session_sizes(sessions_raw: DataFrame) -> DataFrame:
    """Taille de la session de chaque trigger, produit déclencheur exclu.

    Sert de dénominateur identique aux trois bras pour le recall.
    """
    return sessions_raw.select(
        "exec_code",
        "trigger_internal_id",
        F.size(
            F.filter(F.col("session_products"), lambda p: p["internal_id"] != F.col("trigger_internal_id"))
        ).alias("session_size"),
    )


def assemble_arms(prod_results: DataFrame, model_results: DataFrame, sessions_raw: DataFrame) -> DataFrame:
    """Joint les trois bras par exec_code, sur la population de la table prod.

    Un exec_code absent de model_results (trigger hors graphe) donnerait des colonnes null, et
    F.size(null) vaut -1 en Spark : ce qui polluerait silencieusement les moyennes. Le cas est donc
    marqué (missing_model) puis remplacé par des tableaux vides.

    Un trigger sans catégorie ou à pool vide a bien une ligne, mais un top-k vide : il n'a pas
    répondu (model_answered faux), ce qui n'est pas la même chose que répondre sans rien trouver.

    Args:
        prod_results: Table prod du split.
        model_results: Table du modèle.
        sessions_raw: Table sessions_raw du split.

    Returns:
        Une ligne par trigger prod, avec products_returned / positives / negatives de chaque bras,
        session_size, missing_model et model_answered.
    """
    empty_model = F.array().cast(EMPTY_MODEL_TYPE)

    return (
        prod_results
        .select(
            "exec_code",
            "trigger_internal_id",
            *[f"{prefix}_{arm}" for arm in PROD_ARMS for prefix in ("products_returned", "positives", "negatives")],
        )
        .join(
            model_results.select(
                "exec_code",
                F.col("products_returned").alias("products_returned_model"),
                F.col("positives").alias("positives_model"),
                F.col("negatives").alias("negatives_model"),
            ),
            on="exec_code",
            how="left",
        )
        .join(session_sizes(sessions_raw).select("exec_code", "session_size"), on="exec_code", how="left")
        .withColumn("missing_model", F.col("products_returned_model").isNull())
        .withColumns({
            column: F.coalesce(F.col(column), empty_model)
            for column in ("products_returned_model", "positives_model", "negatives_model")
        })
        .withColumn("model_answered", F.size("products_returned_model") > 0)
    )


def add_metrics(df: DataFrame) -> DataFrame:
    """Ajoute par bras le nombre de positifs, found, recall, rang du premier positif et rang réciproque.

    recall est null pour une session vide (session_size <= 0) : il n'y avait rien à retrouver.
    best_rank et rr sont null quand le bras n'a aucun positif.
    """
    for arm in ARMS:
        positives = F.col(f"positives_{arm}")
        df = df.withColumns({
            f"n_positives_{arm}": F.size(positives),
            f"found_{arm}": F.size(positives) > 0,
            f"recall_{arm}": F.when(F.col("session_size") > 0, F.size(positives) / F.col("session_size")),
            f"best_rank_{arm}": F.array_min(F.transform(positives, lambda p: p["rank"])),
        })
        df = df.withColumn(f"rr_{arm}", F.lit(1.0) / F.col(f"best_rank_{arm}"))
    return df
