import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

TOP_K = 12

SORT_BY_RELEVANCE = """
array_sort(
    candidates,
    (a, b) -> CASE
        WHEN a.relevanceScore < b.relevanceScore THEN 1
        WHEN a.relevanceScore > b.relevanceScore THEN -1
        WHEN a.internalId > b.internalId THEN 1
        WHEN a.internalId < b.internalId THEN -1
        ELSE 0
    END
)
"""


def prod_products_by_trigger(df_placements: DataFrame, arr_col: str, top_k: int = TOP_K) -> DataFrame:
    """Fusionne les candidats d'un trigger à travers ses placements et garde les top_k premiers.

    Args:
        df_placements: Placements d'un trigger, une ligne par (executionId,
            placement_pos), avec une colonne de candidats déjà triée
            (arr_col).
        arr_col: Nom de la colonne tableau à utiliser (ex. cands_display,
            cands_relevance).
        top_k: Nombre maximal de candidats conservés par trigger après
            fusion.

    Returns:
        DataFrame avec les colonnes executionId, trigger_internal_id,
        products_returned (liste de structs internal_id/rank). Un candidat
        présent dans plusieurs placements du même trigger est fusionné à
        son meilleur rang (pos_rank minimal) ; le rang final réordonne ces
        candidats fusionnés (tie-break sur internal_id) et ne garde que les
        top_k premiers.
    """
    ranked = (
        df_placements
        .select(
            "executionId",
            "trigger_internal_id",
            F.posexplode(arr_col).alias("candidate_pos", "candidate"),
        )
        .select(
            "executionId",
            "trigger_internal_id",
            F.col("candidate.internalId").cast("bigint").alias("internal_id"),
            (F.col("candidate_pos") + F.lit(1)).alias("pos_rank"),
        )
        .groupBy("executionId", "trigger_internal_id", "internal_id")
        .agg(F.min("pos_rank").alias("pos_rank"))
        .withColumn(
            "rank",
            F.row_number().over(
                Window.partitionBy("executionId", "trigger_internal_id")
                      .orderBy(F.col("pos_rank").asc(), F.col("internal_id").asc())
            ),
        )
        .filter(F.col("rank") <= top_k)
    )

    return (
        ranked
        .groupBy("executionId", "trigger_internal_id")
        .agg(
            F.sort_array(
                F.collect_list(F.struct(F.col("rank"), F.col("internal_id")))
            ).alias("ranked")
        )
        .withColumn(
            "products_returned",
            F.transform(
                "ranked",
                lambda r: F.struct(
                    r["internal_id"].alias("internal_id"),
                    r["rank"].alias("rank"),
                ),
            ),
        )
        .select("executionId", "trigger_internal_id", "products_returned")
    )


def add_pos_neg(df: DataFrame, suffix: str) -> DataFrame:
    """Dérive les listes de positifs et négatifs à partir des produits renvoyés par la production.

    Args:
        df: DataFrame avec les colonnes products_returned_{suffix}
            (candidats renvoyés par la production) et session_ids
            (produits réellement vus dans la session).
        suffix: Variante à traiter ("display" ou "relevance").

    Returns:
        df enrichi de positives_{suffix} et negatives_{suffix} :
        sous-listes de products_returned_{suffix} selon que le produit
        appartient ou non à session_ids. Aucune ligne du DataFrame n'est
        retirée ; contrairement aux négatifs d'entraînement, ces négatifs
        ne sont pas filtrés sur les arêtes du graphe.
    """
    products = F.col(f"products_returned_{suffix}")
    return (
        df
        .withColumn(
            f"positives_{suffix}",
            F.filter(products, lambda p: F.array_contains(F.col("session_ids"), p["internal_id"])),
        )
        .withColumn(
            f"negatives_{suffix}",
            F.filter(products, lambda p: ~F.array_contains(F.col("session_ids"), p["internal_id"])),
        )
    )
