from typing import Any

import pyspark.sql.functions as F
import torch
from pyspark.sql import DataFrame, SparkSession

from fast_gnn_benchmark.data.pipeline.sources import PRODUCT_EMBEDDINGS_TABLE

Trigger = dict[str, Any]


def extract_all_triggers(prod_results: DataFrame, node2idx: dict[int, int]) -> tuple[list[Trigger], int]:
    """Tous les triggers de la table prod, un par exec_code.

    La population d'analyse est définie indépendamment de l'étiquetage : partir de val_split["edge"]
    ne garderait que les exécutions ayant au moins un positif dans le top-12 prod.

    Args:
        prod_results: Table prod, doit contenir exec_code et trigger_internal_id.
        node2idx: Correspondance internalId -> index de nœud.

    Returns:
        Tuple (triggers, skipped) : un dict par exec_code (exec_code, trigger_internal_id,
        trigger_node_id), et le nombre de triggers écartés car leur produit n'est pas dans le graphe.
    """
    rows = prod_results.select("exec_code", "trigger_internal_id").dropDuplicates(["exec_code"]).collect()

    triggers = []
    skipped_no_node = 0

    for row in rows:
        trigger_internal_id = int(row["trigger_internal_id"])

        if trigger_internal_id not in node2idx:
            skipped_no_node += 1
            continue

        triggers.append({
            "exec_code": row["exec_code"],
            "trigger_internal_id": trigger_internal_id,
            "trigger_node_id": node2idx[trigger_internal_id],
        })

    print(f"triggers hors graphe (produit sans node_id): {skipped_no_node}/{len(rows)}")

    return triggers, skipped_no_node


def extract_categories(spark: SparkSession, triggers: list[Trigger], customer_shortname: str) -> int:
    """Attache à chaque trigger, en place, la première t2s_best_fitting_category de son produit.

    Args:
        spark: Session Spark active.
        triggers: Triggers issus de extract_all_triggers.
        customer_shortname: Client dont on lit les catégories.

    Returns:
        Le nombre de triggers sans catégorie (category à None), qui ne seront pas scorés.
    """
    df_trigger_ids = spark.createDataFrame(
        [(str(i),) for i in {t["trigger_internal_id"] for t in triggers}], schema="internalId string"
    )

    rows = (
        spark.table(PRODUCT_EMBEDDINGS_TABLE)
        .filter(F.col("customer_short_name") == customer_shortname)
        .join(F.broadcast(df_trigger_ids), on="internalId", how="left_semi")
        .select("internalId", F.col("t2s_best_fitting_category")[0].alias("category"))
        .dropDuplicates(["internalId"])
        .collect()
    )

    category_by_internal_id = {int(row["internalId"]): row["category"] for row in rows if row["category"] is not None}

    for trigger in triggers:
        trigger["category"] = category_by_internal_id.get(trigger["trigger_internal_id"])

    missing = sum(1 for t in triggers if t["category"] is None)
    print(f"triggers sans categorie: {missing}/{len(triggers)}")

    return missing


def build_candidates_by_category(
    spark: SparkSession, triggers: list[Trigger], node2idx: dict[int, int], customer_shortname: str
) -> dict[str, torch.Tensor]:
    """Construit le pool de candidats de chaque catégorie présente chez les triggers.

    Seuls les produits du graphe sont candidats : un produit sans node_id n'a pas d'embedding GNN.

    Args:
        spark: Session Spark active.
        triggers: Triggers dont la catégorie a été renseignée par extract_categories.
        node2idx: Correspondance internalId -> index de nœud.
        customer_shortname: Client dont on lit le catalogue.

    Returns:
        Dictionnaire catégorie -> tensor long des node_id candidats, sans doublon. Une catégorie
        dont aucun produit n'est dans le graphe a un pool vide.
    """
    unique_categories = {t["category"] for t in triggers if t["category"] is not None}

    rows = (
        spark.table(PRODUCT_EMBEDDINGS_TABLE)
        .filter(
            F.col("t2s_best_fitting_category")[0].isin(list(unique_categories))
            & (F.col("customer_short_name") == customer_shortname)
        )
        .select("internalId", F.col("t2s_best_fitting_category")[0].alias("category"))
        .dropDuplicates(["internalId"])
        .collect()
    )

    internal_ids_by_category: dict[str, list[int]] = {c: [] for c in unique_categories}
    for row in rows:
        internal_ids_by_category[row["category"]].append(int(row["internalId"]))

    # plusieurs internalId peuvent partager le même node_id
    candidates_by_category = {
        category: torch.unique(torch.tensor([node2idx[i] for i in internal_ids if i in node2idx], dtype=torch.long))
        for category, internal_ids in internal_ids_by_category.items()
    }

    pool_sizes = [c.numel() for c in candidates_by_category.values()]
    print(f"nombre de catégories: {len(candidates_by_category)}")
    if pool_sizes:
        print(f"taille moyenne du pool de candidats par catégorie: {sum(pool_sizes) / len(pool_sizes):.0f}")

    return candidates_by_category
