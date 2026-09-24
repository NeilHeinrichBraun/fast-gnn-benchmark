import pandas as pd
import pyspark.sql.functions as F
import torch
from pyspark.sql import DataFrame


def positive_negative_edge_construction(df_recommended: DataFrame, df_split: DataFrame, df_train_edges: DataFrame, node2idx: dict[int, int], split: str) -> tuple:
    """Construit les paires positives et négatives d'un split pour l'entraînement du modèle.

    Args:
        df_recommended: Sortie de recommended_products_for_t2s_user jointe
            aux sessions (colonnes split, userId, session_id, internalId,
            executionId, sponsoredProductPlacementExecutions).
        df_split: Vues sessionisées avec split (colonnes userId, session_id,
            internalId).
        df_train_edges: Arêtes de co-vue d'entraînement, utilisées pour
            écarter les faux négatifs.
        node2idx: Correspondance internalId -> index de nœud.
        split: Split à traiter ("train", "val" ou "test").

    Returns:
        Tuple (pos_edge_index, neg_edge_index, production_mrr_matched,
        production_mrr_coverage, exec2code, code2exec). pos/neg_edge_index
        sont des tensors [3, N] (exec_code, trigger, candidat) ; les deux
        MRR sont calculés sur la population complète des exécutions du
        split, avant tout filtrage sur node2idx. exec2code couvre toutes
        les exécutions du split, indépendamment du label.
    """

    df_recommended_products_t2s_with_sessions = (
        df_recommended
        .filter(F.col("split") == split)
        .select(
            F.col("userId"),
            F.col("session_id"),
            F.col("internalId").alias("internalId_trigger"),
            F.col("executionId"),
            F.col("sponsoredProductPlacementExecutions"),
        )
        .join(
            df_split.filter(F.col("split") == split).select(
                F.col("userId"),
                F.col("session_id"),
                F.col("internalId").alias("internalId_session"),
            ),
            on=["userId", "session_id"],
            how="inner"
        )
    )

    df_recommended_exploded = (
        df_recommended_products_t2s_with_sessions
        .withColumn("placement", F.explode("sponsoredProductPlacementExecutions"))
        .withColumn(
            "top_relevant_products",
            F.slice(F.col("placement.relevantProducts"), 1, 12),
        )
        .select("*", F.posexplode("top_relevant_products").alias("rank_candidate", "candidate"))
        .select(
            F.col("userId"),
            F.col("session_id"),
            F.col("executionId"),
            F.col("internalId_trigger"),
            F.col("internalId_session"),
            F.col("candidate.internalId").alias("internalId_candidate"),
            (F.col("rank_candidate") + F.lit(1)).alias("rank_candidate"),
        )
    ).cache()

    positive_candidates = (
        df_recommended_exploded
        .filter(F.col("internalId_session") == F.col("internalId_candidate"))
        .select(
            F.col("executionId"),
            F.col("internalId_trigger").alias("product_A"),
            F.col("internalId_candidate").alias("product_B"),
        )
        .distinct()
    ).cache()

    matched_rank = (
        df_recommended_exploded
        .filter(F.col("internalId_session") == F.col("internalId_candidate"))
        .groupBy("executionId")
        .agg(F.min("rank_candidate").alias("best_rank"))
        .withColumn("reciprocal_rank", F.lit(1.0) / F.col("best_rank"))
    )

    total_executions = df_recommended_exploded.select("executionId").distinct().count()

    match_stats = matched_rank.agg(
        F.count("*").alias("matched_executions"),
        F.sum("reciprocal_rank").alias("sum_reciprocal_rank"),
    ).collect()[0]
    matched_executions = match_stats["matched_executions"]
    sum_reciprocal_rank = match_stats["sum_reciprocal_rank"] or 0.0

    production_mrr_coverage = sum_reciprocal_rank / total_executions if total_executions > 0 else 0.0

    production_mrr_matched = sum_reciprocal_rank / matched_executions if matched_executions > 0 else 0.0
    coverage = matched_executions / total_executions if total_executions > 0 else 0.0

    print(
        f"[{split}] Production MRR (matched-only, comparable to MRR_trigger): {production_mrr_matched:.4f} | "
        f"Production MRR (coverage-adjusted): {production_mrr_coverage:.4f} | "
        f"{matched_executions}/{total_executions} executions matched (coverage={coverage:.2%})"
    )

    df_all_candidates = (
        df_recommended_exploded
        .select(
            F.col("executionId"),
            F.col("internalId_trigger").alias("product_A"),
            F.col("internalId_candidate").alias("product_B"),
            F.col("rank_candidate")
        )
        .groupBy("executionId", "product_A", "product_B")
        .agg(F.min("rank_candidate").alias("rank_candidate"))
    )

    df_existing_pairs = (
        df_train_edges.select("product_A", "product_B")
        .union(
            df_train_edges.select(
                F.col("product_B").alias("product_A"),
                F.col("product_A").alias("product_B"),
            )
        )
        .distinct()
    )

    df_negative_labeled = (
        df_all_candidates
        .join(
            df_existing_pairs.withColumn("is_existing_edge", F.lit(True)),
            on=["product_A", "product_B"],
            how="left"
        )
        .fillna(False, subset=["is_existing_edge"])
    )

    negative_candidates = (
        df_negative_labeled
        .filter(~F.col("is_existing_edge"))
        .select(
            F.col("executionId"),
            F.col("product_A"),
            F.col("product_B"),
            F.col("rank_candidate"),
        )
        .join(
            positive_candidates.select("executionId", "product_A", "product_B"),
            on=["executionId", "product_A", "product_B"],
            how="left_anti"
        )
    ).cache()

    deepest_pos_rank = (
        df_recommended_exploded
        .filter(F.col("internalId_session") == F.col("internalId_candidate"))
        .groupBy("executionId")
        .agg(F.max("rank_candidate").alias("deepest_pos_rank"))
    )

    exec_with_positives = positive_candidates.select("executionId").distinct()

    negative_candidates_matched = (
        negative_candidates
        .join(F.broadcast(deepest_pos_rank), on="executionId", how="inner")
        .filter(F.col("rank_candidate") <= F.col("deepest_pos_rank"))
        .select("executionId", "product_A", "product_B")
    )

    exec_only_negatives_full = (
        negative_candidates.select("executionId").distinct()
        .join(exec_with_positives, on="executionId", how="left_anti")
    )

    exec_only_negatives = exec_only_negatives_full.sample(fraction=0.0, seed=42)

    negative_candidates_pure_negative = (
        negative_candidates
        .join(F.broadcast(exec_only_negatives), on="executionId", how="left_semi")
        .select("executionId", "product_A", "product_B")
    )

    negative_candidates_sampled = negative_candidates_matched.union(negative_candidates_pure_negative)

    positive_pandas = positive_candidates.toPandas()
    negative_pandas = negative_candidates_sampled.toPandas()

    all_exec_ids = pd.concat([
        positive_pandas["executionId"],
        negative_pandas["executionId"],
        df_recommended.filter(F.col("split") == split).select("executionId").distinct().toPandas()["executionId"],
    ]).unique()
    all_exec_ids.sort()
    exec2code = {eid: i for i, eid in enumerate(all_exec_ids)}
    code2exec = all_exec_ids.tolist()

    positive_pandas["exec_code"] = positive_pandas["executionId"].map(exec2code)
    negative_pandas["exec_code"] = negative_pandas["executionId"].map(exec2code)

    for col in ["product_A", "product_B"]:
        positive_pandas[col] = positive_pandas[col].map(node2idx)
        negative_pandas[col] = negative_pandas[col].map(node2idx)

    positive_pandas = positive_pandas.dropna(subset=["product_A", "product_B", "exec_code"])
    negative_pandas = negative_pandas.dropna(subset=["product_A", "product_B", "exec_code"])

    pos_edge_index = torch.tensor(
        positive_pandas[["exec_code", "product_A", "product_B"]].values.T,
        dtype=torch.long
    )

    neg_edge_index = torch.tensor(
        negative_pandas[["exec_code", "product_A", "product_B"]].values.T,
        dtype=torch.long
    )

    print(f"pos_edge_index shape: {pos_edge_index.shape}")
    print(f"neg_edge_index shape: {neg_edge_index.shape}")

    return (
        pos_edge_index,
        neg_edge_index,
        production_mrr_matched,
        production_mrr_coverage,
        exec2code,
        code2exec,
    )
