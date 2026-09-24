import pyspark.sql.functions as F
from pyspark.sql import DataFrame

USE_SESSION_WEIGHT_NORM = False


def build_co_view_edges(df_sessionized: DataFrame) -> DataFrame:
    """Construit les arêtes de co-vue à partir des vues sessionisées.

    Args:
        df_sessionized: Vues sessionisées (colonnes userId, session_id,
            internalId).

    Returns:
        DataFrame avec les colonnes product_A, product_B, weight, raw_count.
        Une ligne par paire de produits distincts vus dans au moins une même
        session (product_A < product_B) ; weight compte les sessions
        distinctes contenant la paire (ou une somme pondérée par la taille
        de session si USE_SESSION_WEIGHT_NORM est activé), et raw_count le
        nombre brut d'occurrences. Une session à un seul produit distinct ne
        produit aucune arête.
    """
    df_dedup = df_sessionized.select("userId", "session_id", "internalId").distinct()

    session_sizes = (
        df_dedup
        .groupBy("userId", "session_id")
        .agg(F.countDistinct("internalId").alias("session_size"))
    )

    df_dedup = df_dedup.join(session_sizes, on=["userId", "session_id"], how="inner")

    df_pairs = (
        df_dedup.alias("L")
        .join(
            df_dedup.alias("R"),
            on=(
                (F.col("L.userId") == F.col("R.userId"))
                & (F.col("L.session_id") == F.col("R.session_id"))
                & (F.col("L.internalId") < F.col("R.internalId"))
            ),
            how="inner"
        )
        .select(
            F.col("L.internalId").alias("product_A"),
            F.col("R.internalId").alias("product_B"),
            (
                (F.lit(1.0) / (F.col("L.session_size") - F.lit(1)))
                if USE_SESSION_WEIGHT_NORM
                else F.lit(1.0)
            ).alias("pair_weight"),
        )
    )

    return df_pairs.groupBy("product_A", "product_B").agg(
        F.sum("pair_weight").alias("weight"),
        F.count("*").alias("raw_count"),
    )


def remove_train_edges(df_edges: DataFrame, df_train_edges: DataFrame) -> DataFrame:
    """Retire de df_edges les paires déjà présentes dans df_train_edges.

    Args:
        df_edges: Arêtes candidates (val ou test), colonnes product_A,
            product_B.
        df_train_edges: Arêtes d'entraînement à exclure.

    Returns:
        DataFrame identique à df_edges, restreint aux paires absentes de
        df_train_edges. Les paires communes à val et test entre elles ne
        sont pas dédupliquées : une même paire peut se retrouver dans les
        deux.
    """
    return df_edges.join(
        df_train_edges.select("product_A", "product_B"),
        on=["product_A", "product_B"],
        how="left_anti"
    )


def build(df_subset, df_train_edges=None):
    """Construit les arêtes de co-vue d'un sous-ensemble, en retirant celles du train si fourni.

    Args:
        df_subset: Vues sessionisées restreintes à un split (train, val ou
            test).
        df_train_edges: Arêtes d'entraînement à exclure des arêtes
            produites. Laissé à None pour le split train lui-même.

    Returns:
        DataFrame d'arêtes de co-vue (colonnes product_A, product_B,
        weight, raw_count) pour df_subset, débarrassé des paires déjà en
        train si df_train_edges est fourni.
    """
    edges = build_co_view_edges(df_subset)
    if df_train_edges is not None:
        edges = remove_train_edges(edges, df_train_edges)
    return edges


def count_nodes(df_edges: DataFrame) -> int:
    """Compte le nombre de produits distincts apparaissant dans un jeu d'arêtes.

    Args:
        df_edges: Arêtes de co-vue (colonnes product_A, product_B).

    Returns:
        Nombre de valeurs distinctes réunies sur product_A et product_B.
    """
    return (
        df_edges.select(F.col("product_A").alias("node"))
        .union(df_edges.select(F.col("product_B").alias("node")))
        .distinct()
        .count()
    )
