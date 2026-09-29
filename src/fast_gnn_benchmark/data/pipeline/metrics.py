import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def production_mrr_prototype(df: DataFrame, positives_col: str) -> dict[str, float]:
    """MRR de la prod sur la population COMPLETE de triggers.

    matched-only  : moyenne sur les triggers ayant au moins un positif (comparable a MRR_trigger)
    coverage      : les triggers sans positif comptent 0, denominateur = tous les triggers

    Sert aussi pour le modèle : model_results porte une colonne positives de même forme.

    Returns:
        Dictionnaire total, matched, mrr_matched_only, mrr_coverage_adjusted.
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
    summary = {
        "total": total,
        "matched": matched,
        "mrr_matched_only": sum_rr / matched if matched else 0.0,
        "mrr_coverage_adjusted": sum_rr / total if total else 0.0,
    }
    print(
        f"{positives_col:22s} MRR matched-only={summary['mrr_matched_only']:.4f} | "
        f"MRR coverage-adjusted={summary['mrr_coverage_adjusted']:.4f} | "
        f"{matched}/{total} triggers avec au moins un positif ({matched / total if total else 0:.2%})"
    )
    return summary
