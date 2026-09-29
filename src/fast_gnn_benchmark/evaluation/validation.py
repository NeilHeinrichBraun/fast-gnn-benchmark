import pyspark.sql.functions as F
from pyspark.sql import DataFrame

# found_{prod_arm} est constant dans chaque partition : ces cases y sont impossibles.
FORCED_ZERO = {
    "positive": ("model_only", "neither"),
    "full_negative": ("both_find", "prod_only"),
}


def check_alignment(prod_results: DataFrame, model_results: DataFrame) -> int:
    """Vérifie que chaque exec_code désigne le même produit déclencheur côté prod et côté modèle.

    Les triggers du modèle sont extraits de la table prod : l'alignement est garanti par
    construction. Un écart signale deux parquets issus de runs différents du pipeline, dont les
    exec_code ne correspondent pas.

    Args:
        prod_results: Table prod du split.
        model_results: Table du modèle.

    Returns:
        Le nombre d'exec_code comparés.

    Raises:
        ValueError: Si au moins un exec_code porte deux triggers différents.
    """
    row = (
        prod_results.select("exec_code", F.col("trigger_internal_id").alias("tid_prod"))
        .join(
            model_results.select("exec_code", F.col("trigger_internal_id").alias("tid_model")),
            on="exec_code",
            how="inner",
        )
        .agg(
            F.count("*").alias("compared"),
            F.count(F.when(F.col("tid_prod") != F.col("tid_model"), 1)).alias("mismatched"),
        )
        .collect()[0]
    )

    print(f"alignement prod/modele: {row['compared']} exec_code compares, {row['mismatched']} divergents")

    if row["mismatched"]:
        raise ValueError(
            f"{row['mismatched']}/{row['compared']} exec_code divergent entre prod_results et model_results : "
            "parquets issus de deux runs différents du pipeline"
        )

    return row["compared"]


def check_forced_zeros(counts: dict[str, int], session_type: str, label: str) -> None:
    """Vérifie que les cases impossibles d'une matrice par type de session sont bien nulles.

    Args:
        counts: Comptes par case de la matrice.
        session_type: "positive" ou "full_negative".
        label: Nom de la matrice, pour le message d'erreur.

    Raises:
        ValueError: Si une case impossible est non nulle, ce qui signale un partitionnement cassé.
    """
    unexpected = {case: counts[case] for case in FORCED_ZERO[session_type] if counts.get(case, 0)}
    if unexpected:
        raise ValueError(f"{label} : cases censées être nulles dans les sessions {session_type} -> {unexpected}")
