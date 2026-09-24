import pandas as pd


def build_node_mapping(*edge_dfs: pd.DataFrame) -> dict[int, int]:
    """Construit une correspondance internalId -> index contigu (0..N-1).

    Args:
        *edge_dfs: Un ou plusieurs DataFrames d'arêtes (typiquement train,
            val, test), chacun avec les colonnes product_A, product_B.

    Returns:
        Dictionnaire internalId -> index. Le jeu de nœuds est l'union de
        product_A et product_B sur tous les DataFrames fournis ; les index
        sont attribués dans l'ordre croissant des internalId, donc
        déterministes pour un jeu de nœuds donné.
    """

    all_nodes = pd.concat(
        [df["product_A"] for df in edge_dfs] + [df["product_B"] for df in edge_dfs]
    ).unique()

    all_nodes.sort()

    return {node_id: idx for idx, node_id in enumerate(all_nodes)}


def remap_edges(df: pd.DataFrame, node2idx: dict[int, int]) -> pd.DataFrame:
    """Remplace les internalId par leurs index contigus dans un DataFrame d'arêtes.

    Args:
        df: DataFrame d'arêtes (colonnes product_A, product_B) avec des
            internalId bruts.
        node2idx: Correspondance internalId -> index, typiquement
            construite par build_node_mapping sur ce même DataFrame (entre
            autres).

    Returns:
        Copie de df où product_A et product_B contiennent les index
        mappés. Un internalId absent de node2idx devient NaN
        silencieusement (pas d'erreur, pas de ligne retirée) : cette
        fonction suppose que node2idx couvre déjà tous les internalId de
        df.
    """
    df = df.copy()
    df["product_A"] = df["product_A"].map(node2idx)
    df["product_B"] = df["product_B"].map(node2idx)
    return df
