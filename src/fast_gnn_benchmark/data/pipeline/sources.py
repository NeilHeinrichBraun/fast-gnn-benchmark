from datetime import date

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

CUSTOMER_TABLE = "mirakl_ai.ds_etl_prod.t2s_gold_customer"
INSIGHTS_CUSTOMER_TABLE = "mirakl_data_platform.prod_data_platform_silver.artemis_insights_customer"
PUBLISHER_TABLE = "mirakl_data_platform.prod_data_platform_silver.artemis_publisher"
PRODUCT_TABLE = "mirakl_ai.ds_etl_prod.t2s_mongo_product_0_current"
PRODUCT_EMBEDDINGS_TABLE = "mirakl_ai.ds_artemis_prod.product_embeddings"
MERGED_USER_TABLE = "mirakl_data_platform.prod_data_platform_silver.t2s_merged_user"
PRODUCT_DISPLAY_EVENT_TABLE = "mirakl_data_platform.prod_data_platform_silver.t2s_tracking_product_display_page_event_fct"
ADLOG_TABLE = "mirakl_data_platform.prod_data_platform_silver.ads_adlog_fct"

ARTEMIS_NAMESPACE = "artemis-prod"
T2S_NAMESPACE = "target2sell-prod"
EMBEDDING_MODEL_ID = "06bbe5791147457aa395960da021c05c"
ADLOG_PAGE_TYPE = "PRODUCT"
ADLOG_JOIN_TOLERANCE = "INTERVAL 3 SECONDS"
MASTER_PRODUCT_PARENT_ID = -1


def get_customer(spark: SparkSession, customer_shortname: str) -> DataFrame:
    """Résout un client vers son publisherId Artemis.

    Args:
        spark: Session Spark active.
        customer_shortname: shortName du client dans t2s_gold_customer.

    Returns:
        DataFrame avec les colonnes customer_shortname, customerId, db_name,
        publisherId. Un client sans publisher Artemis correspondant (namespace
        artemis-prod) est absent du résultat (inner join).
    """
    customers = (
        spark.table(CUSTOMER_TABLE)
        .where(F.col("shortName") == customer_shortname)
        .select("shortName", "publicId", F.col("databaseName").alias("db_name"))
    )

    publishers = (
        spark.table(INSIGHTS_CUSTOMER_TABLE).alias("I")
        .join(
            spark.table(PUBLISHER_TABLE).alias("P"),
            F.col("P.insights_customer_id") == F.col("I.id"),
            "left",
        )
        .where(F.col("I.__k8s_namespace") == ARTEMIS_NAMESPACE)
        .select(F.col("P.id").alias("publisherId"), F.col("I.public_id").alias("public_id"))
    )

    return (
        customers.join(publishers, customers.publicId == publishers.public_id, "inner")
        .select(
            F.col("shortName").alias("customer_shortname"),
            F.col("publicId").alias("customerId"),
            "db_name",
            "publisherId"
        )
    )


def get_products(spark: SparkSession, db_names: list[str]) -> DataFrame:
    """Charge le catalogue des produits maîtres actifs.

    Args:
        spark: Session Spark active.
        db_names: Bases clients dont on charge le catalogue.

    Returns:
        DataFrame avec les colonnes internalId (bigint), fwProductId, name,
        imageUrl. Une ligne par produit maître (parentId = -1), actif et
        diffusé (isActive, isOkInStream) ; les déclinaisons et les produits
        désactivés ou hors flux sont absents.
    """
    return (
        spark.table(PRODUCT_TABLE)
        .filter(
            F.col("db_name").isin(db_names)
            & F.col("isActive")
            & F.col("isOkInStream")
            & (F.col("parentId") == MASTER_PRODUCT_PARENT_ID)
        )
        .select(
            F.col("internalId").cast("bigint").alias("internalId"),
            "fwProductId",
            "name",
            "imageUrl"
        )
    )


def get_embeddings(spark: SparkSession, db_names: list[str]) -> DataFrame:
    """Charge les embeddings des produits pour une version de modèle fixée.

    Args:
        spark: Session Spark active.
        db_names: Bases clients dont on charge les embeddings.

    Returns:
        DataFrame avec les colonnes internalId, embeddings, last_update_date.
        Une ligne par produit ayant un embedding calculé par le modèle
        EMBEDDING_MODEL_ID fixé dans le module ; les produits embeddés par une
        autre version sont absents.
    """
    return (
        spark.table(PRODUCT_EMBEDDINGS_TABLE)
        .filter(
            F.col("db_name").isin(db_names)
            & (F.col("last_model_id") == EMBEDDING_MODEL_ID)
        )
        .select(
            "internalId",
            "embeddings",
            "last_update_date"
        )
    )


def get_views(spark: SparkSession, customer_ids: list[str], start_date: date, end_date: date) -> DataFrame:
    """Charge les vues produit brutes sur une fenêtre de dates, réconciliées par utilisateur.

    Args:
        spark: Session Spark active.
        customer_ids: Identifiants clients dont on charge les vues.
        start_date: Début de la fenêtre (inclus).
        end_date: Fin de la fenêtre — les bornes sont converties en
            timestamp à minuit, donc la journée de end_date est exclue.

    Returns:
        DataFrame avec les colonnes emitted, internalId, userId. Une ligne
        par vue produit, restreinte aux visiteurs identifiés
        (user.internalId non nul) et résolus en produit maître
        (masterProductInternalId non nul) ; userId est l'identité
        réconciliée via t2s_merged_user quand elle existe.
    """
    df_merged_user = (
        spark.table(MERGED_USER_TABLE)
        .filter(F.col("customerId").isin(customer_ids))
        .select(
            F.col("masterId").alias("userMasterId"),
            F.col("sourceId").alias("userId"),
        )
    )

    return (
        spark.table(PRODUCT_DISPLAY_EVENT_TABLE)
        .filter(
            F.col("customerId").isin(customer_ids)
            & F.col("emitted").between(start_date, end_date)
            & F.col("user.internalId").isNotNull()
            & F.col("masterProductInternalId").isNotNull()
        )
        .withColumnRenamed("masterProductInternalId", "internalId")
        .withColumn("userId", F.col("user.internalId"))
        .join(df_merged_user, on="userId", how="left")
        .withColumn("userId", F.coalesce(F.col("userMasterId"), F.col("userId")))
        .select(
            "emitted",
            "internalId",
            "userId",
        )
    )


def recommended_products_for_t2s_user(
    spark: SparkSession,
    customer_ids: list[str],
    publisher_ids: list[str],
    start_date: date,
    end_date: date,
) -> DataFrame:
    """Associe les logs publicitaires t2s aux vues produit qui les ont suivis.

    Args:
        spark: Session Spark active.
        customer_ids: Identifiants clients concernés.
        publisher_ids: Identifiants publisher Artemis concernés.
        start_date: Début de la fenêtre (inclus).
        end_date: Fin de la fenêtre (bornes à minuit, comme pour get_views).

    Returns:
        DataFrame avec une ligne par vue produit appariée à un log
        publicitaire du même cookie/produit/client, dans une fenêtre de
        ±3 secondes, restreint aux pages de type PRODUCT. Colonnes :
        internalId, emitted, userId (réconcilié via t2s_merged_user),
        customerId, executionId, logInstant, pageType, searchTerm,
        userOrganizeRank, relevantProducts,
        sponsoredProductPlacementExecutions, productsReturned.
    """
    df_merged_user = (
        spark.table(MERGED_USER_TABLE)
        .filter(F.col("customerId").isin(customer_ids))
        .select(
            F.col("masterId").alias("userMasterId"),
            F.col("sourceId").alias("userId"),
        )
    )

    return (
        spark.table(ADLOG_TABLE).alias("adlog")
        .filter(
            (F.col("adlog.__k8s_namespace") == T2S_NAMESPACE)
            & F.col("adlog.publisherId").isin(publisher_ids)
            & F.col("adlog.__log_date").between(start_date, end_date)
            & (F.col("adlog.pageType") == ADLOG_PAGE_TYPE)
        )
        .join(
            spark.table(PRODUCT_DISPLAY_EVENT_TABLE).alias("t2s")
            .filter(
                F.col("t2s.customerId").isin(customer_ids)
                & F.col("t2s.emitted").between(start_date, end_date)
                & F.col("t2s.masterProductInternalId").isNotNull()
                & F.col("t2s.user.internalId").isNotNull()
            ),
            (F.col("adlog.productId") == F.col("t2s.fwProductId"))
            & (F.col("adlog.userCookie") == F.col("t2s.user.cookie"))
            & (F.col("adlog.customerId") == F.col("t2s.customerId"))
            & (F.col("t2s.emitted").between(
                    F.col("adlog.logInstant") - F.expr(ADLOG_JOIN_TOLERANCE),
                    F.col("adlog.logInstant") + F.expr(ADLOG_JOIN_TOLERANCE),
                )),
            how="inner",
        )
        .withColumn("userId", F.col("t2s.user.internalId"))
        .join(df_merged_user, on="userId", how="left")
        .withColumn("userId", F.coalesce(F.col("userMasterId"), F.col("userId")))
        .select(
            F.col("t2s.masterProductInternalId").alias("internalId"),
            F.col("t2s.emitted").alias("emitted"),
            F.col("userId"),
            F.col("t2s.customerId").alias("customerId"),
            F.col("adlog.executionId"),
            F.col("adlog.logInstant"),
            F.col("adlog.pageType"),
            F.col("adlog.searchTerm"),
            F.col("adlog.userOrganizeRank"),
            F.col("adlog.relevantProducts"),
            F.col("adlog.sponsoredProductPlacementExecutions"),
            F.col("adlog.productsReturned"),
        )
    )
