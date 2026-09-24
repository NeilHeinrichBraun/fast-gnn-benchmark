import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def production_mrr_prototype(df: DataFrame, positives_col: str) -> None:
    """MRR de la prod sur la population COMPLETE de triggers.

    matched-only  : moyenne sur les triggers ayant au moins un positif (comparable a MRR_trigger)
    coverage      : les triggers sans positif comptent 0, denominateur = tous les triggers
    """
    row = (
        df
        .select(F.array_min(F.transform(positives_col, lambda p: p["rank"])).alias("best_rank"))
        .agg(
            F.count("*").alias("total"),
            F.count("best_rank").alias("matched"),
            F.sum(F.lit(1.0) / F.col("best_rank")).alias("sum_rr"),
        )
        .collect()[0]
    )
    total, matched = row["total"], row["matched"]
    sum_rr = row["sum_rr"] or 0.0
    print(
        f"{positives_col:22s} MRR matched-only={sum_rr / matched if matched else 0:.4f} | "
        f"MRR coverage-adjusted={sum_rr / total if total else 0:.4f} | "
        f"{matched}/{total} triggers avec au moins un positif ({matched / total:.2%})"
    )
