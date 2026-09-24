from datetime import date

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window


def sessionize_views(df_views: DataFrame, session_timeout: int = 1800) -> DataFrame:
    """Découpe les vues d'un utilisateur en sessions par expiration d'inactivité.

    Args:
        df_views: Vues par utilisateur (colonnes userId, internalId, emitted).
        session_timeout: Durée d'inactivité en secondes au-delà de laquelle
            une nouvelle session démarre.

    Returns:
        DataFrame avec les colonnes userId, session_id, internalId, emitted,
        session_start. Une ligne par vue, enrichie de son identifiant de
        session et de l'horodatage de début de session ; aucune vue n'est
        retirée.
    """
    user_window = Window.partitionBy("userId").orderBy("emitted", "internalId")

    df = df_views.withColumn(
        "seconds_since_last",
        (F.col("emitted") - F.lag("emitted").over(user_window)).cast("long")
    )

    df = df.withColumn(
        "is_new_session",
        F.when(
            F.col("seconds_since_last").isNull() | (F.col("seconds_since_last") > session_timeout),
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    df = df.withColumn(
        "session_id",
        F.sum("is_new_session").over(user_window.rowsBetween(Window.unboundedPreceding, Window.currentRow))
    )

    session_window = Window.partitionBy("userId", "session_id")
    df = df.withColumn(
        "session_start",
        F.min("emitted").over(session_window)
    )

    return df.select("userId", "session_id", "internalId", "emitted", "session_start")


def split_sessions(df_sessionized: DataFrame, cutoff_train: date, cutoff_val: date) -> DataFrame:
    """Assigne chaque session à train/val/test selon deux coupures de date.

    Args:
        df_sessionized: Vues sessionisées (colonnes userId, session_id,
            session_start, ...).
        cutoff_train: Date avant laquelle une session part en train.
        cutoff_val: Date avant laquelle (et après cutoff_train) une session
            part en val ; au-delà, elle part en test.

    Returns:
        DataFrame identique à df_sessionized, enrichi d'une colonne split
        (train/val/test). La coupure se fait sur session_start : toutes les
        vues d'une même session tombent du même côté, même si la session
        déborde sur une autre période.
    """
    df_sessions = df_sessionized.select("userId", "session_id", "session_start").distinct()

    df_sessions = df_sessions.withColumn(
        "split",
        F.when(F.col("session_start") < F.lit(cutoff_train), F.lit("train"))
        .when(F.col("session_start") < F.lit(cutoff_val), F.lit("val"))
        .otherwise(F.lit("test"))
    )

    return df_sessionized.join(
        df_sessions.select("userId", "session_id", "split"),
        on=["userId", "session_id"],
        how="inner",
    )
