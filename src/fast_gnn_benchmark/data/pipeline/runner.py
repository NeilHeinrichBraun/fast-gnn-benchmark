from functools import reduce
from typing import Any, NamedTuple

import pandas as pd
import pyspark.sql.functions as F
import yaml
from pyspark.sql import DataFrame, SparkSession
from torch_geometric.data import Data

from fast_gnn_benchmark.data.pipeline import (
    artifacts,
    candidates,
    coview_edges,
    features,
    graph,
    mapping,
    metrics,
    prod_tables,
    sessions,
    sources,
    tensors,
    validation,
)
from fast_gnn_benchmark.schemas.pipeline_model import GraphPipelineParameters, Split

SPLITS: tuple[Split, ...] = ("train", "val", "test")
VARIANTS = ("display", "relevance")
EMPTY_PRODUCTS_TYPE = "array<struct<internal_id:bigint,rank:int>>"


class CustomerReference(NamedTuple):
    """Identifiants d'un client. Un client peut couvrir plusieurs bases et plusieurs publishers."""

    customer_ids: list[str]
    publisher_ids: list[str]
    db_names: list[str]


def get_pipeline_parameters_from_config(config_file: str) -> GraphPipelineParameters:
    """Charge et valide une configuration de pipeline.

    Args:
        config_file: Chemin du YAML de configuration.

    Returns:
        La configuration validée.
    """
    with open(config_file, "r") as file:
        return GraphPipelineParameters.model_validate(yaml.safe_load(file))


def resolve_customer(spark: SparkSession, customer_shortname: str) -> CustomerReference:
    """Collecte les identifiants du client sur le driver.

    Args:
        spark: Session Spark active.
        customer_shortname: shortName du client.

    Returns:
        Les trois listes d'identifiants du client.

    Raises:
        ValueError: Si aucun publisher Artemis ne correspond au client demandé.
    """
    rows = sources.get_customer(spark, customer_shortname).collect()

    if not rows:
        raise ValueError(f"Aucun publisher Artemis pour le client {customer_shortname!r}")

    return CustomerReference(
        customer_ids=[row["customerId"] for row in rows],
        publisher_ids=[row["publisherId"] for row in rows],
        db_names=[row["db_name"] for row in rows],
    )


def graph_nodes(edge_dfs: list[DataFrame]) -> DataFrame:
    """Réunit les produits apparaissant dans un ou plusieurs jeux d'arêtes.

    Args:
        edge_dfs: Jeux d'arêtes (colonnes product_A, product_B).

    Returns:
        DataFrame à une colonne internalId, sans doublon.
    """
    sides = [
        df.select(F.col(column).alias("internalId"))
        for df in edge_dfs
        for column in ("product_A", "product_B")
    ]
    return reduce(DataFrame.union, sides).distinct()


def build_prod_tables(
    spark: SparkSession,
    df_recommended: DataFrame,
    df_split: DataFrame,
    exec2code: dict[Any, int],
    split: Split,
    top_k: int,
) -> tuple[DataFrame, DataFrame]:
    """Construit les deux tables de comparaison avec la production pour un split.

    Args:
        spark: Session Spark active.
        df_recommended: Logs publicitaires joints aux sessions.
        df_split: Vues sessionisées avec leur split.
        exec2code: Correspondance executionId -> exec_code du split.
        split: Split à traiter.
        top_k: Profondeur du classement prod conservée par trigger.

    Returns:
        Tuple (sessions_raw, prod_results). sessions_raw porte les produits réellement vus dans
        la session de chaque trigger ; prod_results porte les produits renvoyés par la prod dans
        les deux variantes, avec leurs positifs et négatifs dérivés.
    """
    df_recommended_split = df_recommended.filter(F.col("split") == split)

    triggers = df_recommended_split.select(
        "userId", "session_id", "executionId", F.col("internalId").alias("trigger_internal_id")
    ).distinct()

    session_products = (
        triggers
        .join(
            df_split.filter(F.col("split") == split).select("userId", "session_id", "internalId", "emitted"),
            on=["userId", "session_id"],
            how="inner",
        )
        .groupBy("userId", "session_id", "executionId", "trigger_internal_id")
        .agg(
            F.collect_list(
                F.struct(F.col("internalId").alias("internal_id"), F.col("emitted"))
            ).alias("session_products")
        )
    )

    df_exec2code = spark.createDataFrame(pd.DataFrame(exec2code.items(), columns=["executionId", "exec_code"]))

    sessions_raw = (
        session_products
        .join(df_exec2code, on="executionId", how="inner")
        .select(
            "exec_code",
            "trigger_internal_id",
            F.col("executionId").alias("execution_id"),
            "session_id",
            "session_products",
        )
    ).cache()

    n_triggers, n_sessions = triggers.count(), sessions_raw.count()
    print(f"{split}_triggers: {n_triggers} triggers")
    print(f"sessions_raw_{split}: {n_sessions} triggers (perdus par le join exec_code: {n_triggers - n_sessions})")
    validation.check_counts_match(n_sessions, n_triggers, f"sessions_raw_{split}")
    sessions_raw.printSchema()

    placements = (
        df_recommended_split
        .select(
            F.col("executionId"),
            F.col("internalId").alias("trigger_internal_id"),
            F.posexplode("sponsoredProductPlacementExecutions").alias("placement_pos", "placement"),
        )
        .select(
            "executionId",
            "trigger_internal_id",
            "placement_pos",
            F.col("placement.relevantProducts").alias("candidates"),
        )
        .dropDuplicates(["executionId", "placement_pos"])
        .withColumn("cands_display", F.slice(F.col("candidates"), 1, top_k))
        .withColumn("cands_relevance", F.slice(F.expr(prod_tables.SORT_BY_RELEVANCE), 1, top_k))
    ).cache()

    print(f"placements_{split}: {placements.count()} placements")

    prod_results = (
        sessions_raw
        .select(
            "exec_code",
            "trigger_internal_id",
            F.col("execution_id").alias("executionId"),
            F.col("session_products.internal_id").alias("session_ids"),
        )
        .dropDuplicates(["exec_code"])
        .withColumn("session_ids", F.coalesce(F.col("session_ids"), F.array().cast("array<bigint>")))
    )

    for variant in VARIANTS:
        products = prod_tables.prod_products_by_trigger(placements, f"cands_{variant}", top_k)
        print(f"{variant}: {products.count()} triggers")
        column = f"products_returned_{variant}"
        prod_results = (
            prod_results
            .join(
                products.withColumnRenamed("products_returned", column),
                on=["executionId", "trigger_internal_id"],
                how="left",
            )
            .withColumn(column, F.coalesce(F.col(column), F.array().cast(EMPTY_PRODUCTS_TYPE)))
        )
        prod_results = prod_tables.add_pos_neg(prod_results, variant)

    prod_results = prod_results.select(
        "exec_code",
        "trigger_internal_id",
        F.col("executionId").alias("execution_id"),
        *[
            f"{prefix}_{variant}"
            for variant in VARIANTS
            for prefix in ("products_returned", "positives", "negatives")
        ],
    ).cache()

    print(f"prod_results_{split}: {prod_results.count()} triggers")
    prod_results.printSchema()

    for variant in VARIANTS:
        metrics.production_mrr_prototype(prod_results, f"positives_{variant}")

    return sessions_raw, prod_results


def do_build(spark: SparkSession, parameters: GraphPipelineParameters) -> Data:
    """Construit le graphe de co-vue et sauvegarde tous les artefacts sur S3.

    Args:
        spark: Session Spark active.
        parameters: Configuration du pipeline.

    Returns:
        L'objet PyG Data écrit sur S3.
    """
    customer = resolve_customer(spark, parameters.customer_shortname)
    print(
        f"Client {parameters.customer_shortname}: {len(customer.customer_ids)} customerId, "
        f"{len(customer.publisher_ids)} publisherId, {len(customer.db_names)} db_name"
    )
    print(
        f"Fenêtre {parameters.start_date} -> {parameters.end_date} | "
        f"cutoff train {parameters.cutoff_train} | cutoff val {parameters.cutoff_val}"
    )

    df_products = sources.get_products(spark, customer.db_names)
    df_embeddings = sources.get_embeddings(spark, customer.db_names)
    df_views = sources.get_views(spark, customer.customer_ids, parameters.start_date, parameters.end_date)

    df_sessionized = sessions.sessionize_views(
        df_views.join(df_products, on="internalId", how="inner"), parameters.session_timeout
    )
    df_split = sessions.split_sessions(df_sessionized, parameters.cutoff_train, parameters.cutoff_val).cache()

    df_train_edges = coview_edges.build(df_split.filter(F.col("split") == "train"))
    df_val_edges = coview_edges.build(df_split.filter(F.col("split") == "val"), df_train_edges)
    df_test_edges = coview_edges.build(df_split.filter(F.col("split") == "test"), df_train_edges)
    edge_dfs = [df_train_edges, df_val_edges, df_test_edges]

    df_graph_nodes = graph_nodes(edge_dfs)
    edge_counts = dict(zip(SPLITS, [df.count() for df in edge_dfs], strict=True))

    print(f"Total nodes: {df_graph_nodes.count()}")
    for split, count in edge_counts.items():
        print(f"{split:6s} — edges: {count}")

    edges_pandas = [df.toPandas() for df in edge_dfs]
    emb_pandas = df_embeddings.join(df_graph_nodes, on="internalId", how="inner").toPandas()

    node2idx = mapping.build_node_mapping(*edges_pandas)
    num_nodes = len(node2idx)
    validation.check_num_nodes(num_nodes, parameters.expected_num_nodes)
    print(f"Number of nodes: {num_nodes}")

    train_remapped = mapping.remap_edges(edges_pandas[0], node2idx)
    validation.check_remapped_edges(train_remapped, "train_edges")
    train_edge_index, train_edge_attr = tensors.edges_to_tensors(train_remapped, parameters.symmetric_edges)
    print(f"Train edges (symmetric): {train_edge_index.shape[1]}")
    print(f"Train edge_index shape: {tuple(train_edge_index.shape)}")
    print(f"Train edge_attr shape: {tuple(train_edge_attr.shape)}")

    x = features.build_node_features(emb_pandas, node2idx)
    nodes_without_embedding = validation.check_node_features(x, parameters.max_nodes_without_embedding_ratio)
    print(f"x shape: {tuple(x.shape)}")
    print(f"Nodes without embedding: {nodes_without_embedding}")

    df_recommended = df_split.join(
        sources.recommended_products_for_t2s_user(
            spark, customer.customer_ids, customer.publisher_ids, parameters.start_date, parameters.end_date
        ),
        on=["internalId", "emitted", "userId"],
        how="inner",
    ).cache()

    labels = {
        split: candidates.positive_negative_edge_construction(
            df_recommended, df_split, df_train_edges, node2idx, split
        )
        for split in SPLITS
    }

    for split, (pos, neg, *_) in labels.items():
        validation.check_edge_tensor(pos, f"{split}_pos_edge_index", num_nodes)
        validation.check_edge_tensor(neg, f"{split}_neg_edge_index", num_nodes)

    data = graph.build_pyg_data(
        num_nodes=num_nodes,
        train_edge_index=train_edge_index,
        train_edge_attr=train_edge_attr,
        train_pos_edge_index=labels["train"][0],
        train_neg_edge_index=labels["train"][1],
        val_pos_edge_index=labels["val"][0],
        val_neg_edge_index=labels["val"][1],
        test_pos_edge_index=labels["test"][0],
        test_neg_edge_index=labels["test"][1],
        x=x,
    )

    print(data)

    exec_mappings = {
        split: {
            "exec2code": {str(execution_id): code for execution_id, code in labels[split][4].items()},
            "code2exec": labels[split][5],
        }
        for split in SPLITS
    }

    print(f"{artifacts.save_graph(data, parameters.artifacts)} uploaded")
    print(f"{artifacts.save_node_mapping(node2idx, parameters.artifacts)} uploaded")
    print(f"{artifacts.save_exec_mappings(exec_mappings, parameters.artifacts)} uploaded")

    for split in parameters.prod_table_splits:
        sessions_raw, prod_results = build_prod_tables(
            spark, df_recommended, df_split, labels[split][4], split, parameters.top_k
        )
        for df, filename in (
            (sessions_raw, parameters.artifacts.sessions_raw_filename(split)),
            (prod_results, parameters.artifacts.prod_results_filename(split)),
        ):
            print(f"{artifacts.write_parquet(df, filename, parameters.artifacts)} uploaded")

    manifest = {
        "parameters": parameters.model_dump(mode="json"),
        "num_nodes": num_nodes,
        "nodes_without_embedding": nodes_without_embedding,
        "embedding_dim": x.shape[1],
        "co_view_edges": edge_counts,
        "labelled_pairs": {
            split: {"positive": labels[split][0].shape[1], "negative": labels[split][1].shape[1]}
            for split in SPLITS
        },
        "production_mrr": {
            split: {"matched": labels[split][2], "coverage": labels[split][3]} for split in SPLITS
        },
    }

    print(f"{artifacts.save_manifest(manifest, parameters.artifacts)} uploaded")

    return data
