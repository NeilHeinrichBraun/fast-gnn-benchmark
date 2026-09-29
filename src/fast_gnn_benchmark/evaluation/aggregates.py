"""Statistiques agrégées sur la table par trigger de assembly.add_metrics.

Chaque fonction renvoie un dictionnaire sérialisable en JSON et imprime sa lecture dans les logs.

Le MRR de référence est le coverage-adjusted : la population n'est pas conditionnée à un succès de
la prod, donc les triggers où personne ne trouve rien comptent zéro. Le matched-only se lit « quand
ce bras trouve, à quel rang ».
"""

from typing import Any

import pyspark.sql.functions as F
from pyspark.sql import Column, DataFrame

from fast_gnn_benchmark.evaluation import validation
from fast_gnn_benchmark.evaluation.assembly import ARMS, PROD_ARMS

CASES = ("both_find", "prod_only", "model_only", "neither")
SESSION_TYPES = ("positive", "full_negative")
VERDICTS = ("les_deux_trouvent", "display_seul", "relevance_seul", "aucun_ne_trouve")


def _pct(count: int, total: int) -> float:
    return round(count / total * 100, 2) if total else 0.0


def _case(prod_arm: str) -> Column:
    model, prod = F.col("found_model"), F.col(f"found_{prod_arm}")
    return (
        F.when(model & prod, "both_find")
        .when(model & ~prod, "model_only")
        .when(~model & prod, "prod_only")
        .otherwise("neither")
    )


def _case_counts(df: DataFrame, prod_arm: str) -> dict[str, int]:
    """Compte les triggers par case modèle x prod_{prod_arm}, en une seule action (4 lignes au plus)."""
    rows = df.withColumn("case", _case(prod_arm)).groupBy("case").count().collect()
    counts = {row["case"]: row["count"] for row in rows}
    return {case: counts.get(case, 0) for case in CASES}


def _matrix(counts: dict[str, int]) -> dict[str, dict[str, float]]:
    total = sum(counts.values())
    return {case: {"count": count, "pct": _pct(count, total)} for case, count in counts.items()}


def _print_matrix(matrix: dict[str, dict[str, float]]) -> None:
    for case, cell in matrix.items():
        print(f"   {case:12s} {cell['count']:>9d} {cell['pct']:>7.2f}%")


def aggregate_metrics(df: DataFrame) -> dict[str, Any]:
    """recall, taux de found, positifs moyens et MRR (coverage-adjusted et matched-only) par bras.

    Args:
        df: Table par trigger avec les métriques de add_metrics.

    Returns:
        Dictionnaire total_triggers, recall_undefined, et une entrée par bras.
    """
    exprs = [
        F.count("*").alias("total_triggers"),
        F.count(F.when(F.col("recall_model").isNull(), 1)).alias("recall_undefined"),
    ]
    for arm in ARMS:
        exprs += [
            F.avg(f"recall_{arm}").alias(f"avg_recall_{arm}"),
            F.avg(F.col(f"found_{arm}").cast("int")).alias(f"pct_found_{arm}"),
            F.avg(f"n_positives_{arm}").alias(f"avg_n_positives_{arm}"),
            F.sum(f"rr_{arm}").alias(f"sum_rr_{arm}"),
            F.count(f"best_rank_{arm}").alias(f"matched_{arm}"),
        ]

    row = df.agg(*exprs).collect()[0]
    total = row["total_triggers"]

    arms = {}
    for arm in ARMS:
        matched = row[f"matched_{arm}"]
        sum_rr = row[f"sum_rr_{arm}"] or 0.0
        arms[arm] = {
            "avg_recall": row[f"avg_recall_{arm}"] or 0.0,
            "pct_found": row[f"pct_found_{arm}"] or 0.0,
            "avg_n_positives": row[f"avg_n_positives_{arm}"] or 0.0,
            "mrr_coverage_adjusted": sum_rr / total if total else 0.0,
            "mrr_matched_only": sum_rr / matched if matched else 0.0,
            "matched": matched,
        }

    print(f"triggers: {total}")
    print(f"recall non defini (session vide): {row['recall_undefined']}")
    print()
    header = f"{'bras':12s} {'recall':>9s} {'found':>8s} {'n_pos':>7s} {'MRR cov':>9s} {'MRR match':>10s} {'matched':>9s}"
    print(header)
    print("-" * len(header))
    for arm, m in arms.items():
        print(
            f"{arm:12s} {m['avg_recall']:9.4f} {m['pct_found']:8.2%} {m['avg_n_positives']:7.3f} "
            f"{m['mrr_coverage_adjusted']:9.4f} {m['mrr_matched_only']:10.4f} {m['matched']:9d}"
        )

    return {"total_triggers": total, "recall_undefined": row["recall_undefined"], "arms": arms}


def matrix_2x2(df: DataFrame, prod_arm: str) -> dict[str, Any]:
    """Matrice modèle x prod_{prod_arm} sur les triggers auxquels le modèle a répondu.

    Les triggers sans réponse du modèle (pas de catégorie ou pool vide) sont exclus et comptés à
    part : les confondre avec « a répondu sans rien trouver » gonflerait la case prod_only.

    Args:
        df: Table par trigger avec les métriques de add_metrics.
        prod_arm: "display" ou "relevance".

    Returns:
        Dictionnaire valid, excluded et cases (count et pct par case).
    """
    excluded = df.filter(~F.col("model_answered")).count()
    matrix = _matrix(_case_counts(df.filter(F.col("model_answered")), prod_arm))
    valid = sum(cell["count"] for cell in matrix.values())

    print(f"=== modele vs prod_{prod_arm} — {valid} triggers valides ({excluded} exclus: modele n'a pas repondu) ===")
    _print_matrix(matrix)

    return {"valid": valid, "excluded": excluded, "cases": matrix}


def product_overlap(df: DataFrame) -> dict[str, float]:
    """Taille moyenne des listes et recouvrement moyen entre bras.

    pct_display_eq_relevance proche de 1 signifie que les deux variantes prod renvoient les mêmes
    produits : les deux matrices modèle x prod doivent alors être quasi identiques.
    """
    ids = lambda column: F.transform(column, lambda p: p["internal_id"])  # noqa: E731
    overlap = lambda a, b: F.size(F.array_intersect(ids(f"products_returned_{a}"), ids(f"products_returned_{b}")))  # noqa: E731

    row = (
        df
        .select(
            *[F.size(f"products_returned_{arm}").alias(f"n_{arm}") for arm in ARMS],
            overlap("display", "relevance").alias("ov_display_relevance"),
            overlap("model", "display").alias("ov_model_display"),
            overlap("model", "relevance").alias("ov_model_relevance"),
        )
        .agg(
            F.count("*").alias("total_triggers"),
            *[F.avg(f"n_{arm}").alias(f"avg_n_{arm}") for arm in ARMS],
            *[F.avg(c).alias(f"avg_{c}") for c in ("ov_display_relevance", "ov_model_display", "ov_model_relevance")],
            F.avg(
                ((F.col("ov_display_relevance") == F.col("n_display")) & (F.col("n_display") == F.col("n_relevance")))
                .cast("int")
            ).alias("pct_display_eq_relevance"),
        )
        .collect()[0]
        .asDict()
    )

    for key, value in row.items():
        print(f"{key:28s} {value}")

    return row


def by_prod_session(df: DataFrame, prod_arm: str) -> dict[str, dict[str, float]]:
    """Scores du modèle et de prod_{prod_arm}, par type de session selon cette variante.

    positive : la prod a trouvé au moins un produit vu ; full_negative : elle n'a rien trouvé. Dans
    full_negative, les métriques de la variante sont nulles par construction ; l'information est
    celle du modèle, son taux de rattrapage là où la prod est passée à côté.

    Les sessions vides et les triggers sans réponse du modèle sont écartés.

    Args:
        df: Table par trigger avec les métriques de add_metrics.
        prod_arm: "display" ou "relevance".

    Returns:
        Dictionnaire type de session -> métriques.
    """
    rows = (
        df
        .filter((F.col("session_size") > 0) & F.col("model_answered"))
        .withColumn("prod_session", F.when(F.col(f"found_{prod_arm}"), "positive").otherwise("full_negative"))
        .groupBy("prod_session")
        .agg(
            F.count("*").alias("triggers"),
            F.avg(F.col("found_model").cast("int")).alias("pct_model_found"),
            F.avg("recall_model").alias("avg_recall_model"),
            F.avg(f"recall_{prod_arm}").alias(f"avg_recall_{prod_arm}"),
            (F.coalesce(F.sum("rr_model"), F.lit(0.0)) / F.count("*")).alias("mrr_model_cov"),
            (F.coalesce(F.sum(f"rr_{prod_arm}"), F.lit(0.0)) / F.count("*")).alias(f"mrr_{prod_arm}_cov"),
        )
        .collect()
    )

    result = {row["prod_session"]: {k: v for k, v in row.asDict().items() if k != "prod_session"} for row in rows}

    print(f"===== type de session selon prod_{prod_arm} =====")
    for session_type in SESSION_TYPES:
        print(f"   {session_type:14s} {result.get(session_type)}")

    return result


def variant_crossing(df: DataFrame) -> dict[str, Any]:
    """Croise le verdict des deux variantes prod : le choix des douze produits change-t-il le résultat ?

    Les cases hors diagonale sont le cœur du sujet : display_seul compte les triggers où l'ordre
    d'affichage fait remonter un produit vu que le tri par relevanceScore aurait manqué.

    Args:
        df: Table par trigger avec les métriques de add_metrics.

    Returns:
        Dictionnaire total (sessions non vides) et verdicts (triggers et pct par verdict).
    """
    display, relevance = F.col("found_display"), F.col("found_relevance")
    verdict = (
        F.when(display & relevance, "les_deux_trouvent")
        .when(display & ~relevance, "display_seul")
        .when(~display & relevance, "relevance_seul")
        .otherwise("aucun_ne_trouve")
    )

    rows = df.filter(F.col("session_size") > 0).withColumn("verdict", verdict).groupBy("verdict").count().collect()
    counts = {row["verdict"]: row["count"] for row in rows}
    total = sum(counts.values())
    verdicts = {v: {"triggers": counts.get(v, 0), "pct": _pct(counts.get(v, 0), total)} for v in VERDICTS}

    print(f"triggers avec session non vide: {total}")
    for v, cell in verdicts.items():
        print(f"   {v:18s} {cell['triggers']:>9d} {cell['pct']:>7.2f}%")

    return {"total": total, "verdicts": verdicts}


def matrices_by_session(df: DataFrame) -> dict[str, dict[str, Any]]:
    """Quatre matrices modèle x prod : deux variantes x deux types de session définis par la variante.

    Deux des quatre cases de chaque matrice sont structurellement nulles ; elles servent de contrôle
    et un écart fait échouer le job. L'information est la répartition entre les deux autres, soit le
    taux de réussite du modèle dans ce régime.

    Args:
        df: Table par trigger avec les métriques de add_metrics.

    Returns:
        Dictionnaire "{prod_arm}_{session_type}" -> matrice (total et cases).
    """
    base = df.filter(F.col("model_answered") & (F.col("session_size") > 0))
    matrices = {}

    for prod_arm in PROD_ARMS:
        for session_type in SESSION_TYPES:
            found = F.col(f"found_{prod_arm}")
            counts = _case_counts(base.filter(found if session_type == "positive" else ~found), prod_arm)
            label = f"modele vs prod_{prod_arm} | sessions {session_type}"
            validation.check_forced_zeros(counts, session_type, label)

            matrix = _matrix(counts)
            total = sum(counts.values())
            print(f"=== {label} --- {total} triggers ===")
            _print_matrix(matrix)

            matrices[f"{prod_arm}_{session_type}"] = {"total": total, "cases": matrix}

    return matrices


def session_level(sessions_raw: DataFrame, df: DataFrame) -> dict[str, Any]:
    """Répartition des sessions (et non plus des triggers) selon chaque variante prod.

    sessions_raw ne porte pas userId et session_id n'est qu'un compteur par utilisateur : la clé
    est reconstruite à partir du contenu de la session. Les identifiants sont triés avant de hacher,
    collect_list n'ordonnant pas les triggers d'une même session de la même façon. C'est un proxy de
    (userId, session_id), perdu dans le select final du pipeline.

    Une session est positive dès qu'au moins un de ses triggers trouve un produit vu (F.max).

    Args:
        sessions_raw: Table sessions_raw du split.
        df: Table par trigger avec les métriques de add_metrics.

    Returns:
        Dictionnaire des comptes, de la répartition par variante et du croisement display x relevance.
    """
    session_key = F.xxhash64(
        F.col("session_id"),
        F.sort_array(F.transform("session_products", lambda p: p["internal_id"])),
        F.array_min(F.transform("session_products", lambda p: p["emitted"])),
    )

    trigger_to_session = sessions_raw.select(
        "exec_code",
        session_key.alias("session_key"),
        F.size("session_products").alias("n_session_products"),
    )

    sessions = (
        trigger_to_session
        .join(df.select("exec_code", "found_display", "found_relevance", "found_model"), on="exec_code", how="inner")
        .groupBy("session_key")
        .agg(
            F.count("*").alias("n_triggers"),
            F.max("n_session_products").alias("n_session_products"),
            F.max(F.col("found_display").cast("int")).alias("any_display"),
            F.max(F.col("found_relevance").cast("int")).alias("any_relevance"),
            F.max(F.col("found_model").cast("int")).alias("any_model"),
            F.sum(F.col("found_display").cast("int")).alias("n_found_display"),
            F.sum(F.col("found_relevance").cast("int")).alias("n_found_relevance"),
        )
        .withColumns({
            f"sess_{arm}": F.when(F.col(f"any_{arm}") == 1, "positive").otherwise("full_negative")
            for arm in PROD_ARMS
        })
    ).cache()

    n_sessions = sessions.count()
    n_triggers = trigger_to_session.count()

    print(f"triggers (= executions prod) : {n_triggers}")
    print(f"sessions (proxy de cle)      : {n_sessions}")
    print(f"triggers par session         : {n_triggers / n_sessions if n_sessions else 0:.2f}")

    by_arm = {}
    for arm in PROD_ARMS:
        rows = (
            sessions
            .groupBy(f"sess_{arm}")
            .agg(
                F.count("*").alias("sessions"),
                F.avg("n_triggers").alias("triggers_moy"),
                F.avg("n_session_products").alias("produits_vus_moy"),
                F.avg(f"n_found_{arm}").alias("triggers_trouvant_moy"),
                F.avg("any_model").alias("pct_model_found"),
            )
            .collect()
        )
        by_arm[arm] = {
            row[f"sess_{arm}"]: {
                **{k: v for k, v in row.asDict().items() if k != f"sess_{arm}"},
                "pct": _pct(row["sessions"], n_sessions),
            }
            for row in rows
        }
        print(f"===== repartition des SESSIONS selon prod_{arm} =====")
        for session_type in SESSION_TYPES:
            print(f"   {session_type:14s} {by_arm[arm].get(session_type)}")

    crossed_rows = sessions.groupBy("sess_display", "sess_relevance").count().collect()
    crossed = {d: dict.fromkeys(SESSION_TYPES, 0) for d in SESSION_TYPES}
    for row in crossed_rows:
        crossed[row["sess_display"]][row["sess_relevance"]] = row["count"]

    print("===== croisement display (lignes) x relevance (colonnes), au niveau SESSION =====")
    for d in SESSION_TYPES:
        print(f"   {d:14s} {crossed[d]}")

    sessions.unpersist()

    return {"triggers": n_triggers, "sessions": n_sessions, "by_prod_arm": by_arm, "display_x_relevance": crossed}
