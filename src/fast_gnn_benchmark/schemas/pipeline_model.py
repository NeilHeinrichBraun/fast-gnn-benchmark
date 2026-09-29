from datetime import date, timedelta
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Split = Literal["train", "val", "test"]


class ArtifactParameters(BaseModel):
    """Emplacement et nommage des artefacts produits.

    suffix isole un run des artefacts partagés : les exec_code diffusent dans tous les parquets,
    donc écraser les chemins d'un run précédent casserait model_test et model_stat, qui lisent
    encore l'encodage de ce run.
    """

    model_config = ConfigDict(extra="forbid")

    bucket: str = "mirakl-data-science-tmp2"
    prefix: str = "nbraun/datasets/coview-mdm"
    suffix: str = ""
    staging_dir: str = "/dbfs/tmp/nbraun/datasets/coview-mdm"

    def key(self, filename: str) -> str:
        return f"{self.prefix.strip('/')}/{filename}"

    def uri(self, filename: str) -> str:
        return f"s3://{self.bucket}/{self.key(filename)}"

    def staging_path(self, filename: str) -> str:
        return f"{self.staging_dir.rstrip('/')}/{filename}"

    @property
    def graph_filename(self) -> str:
        return f"data{self.suffix}.pt"

    @property
    def node_mapping_filename(self) -> str:
        return f"node2idx{self.suffix}.json"

    @property
    def exec_mappings_filename(self) -> str:
        return f"exec_mappings{self.suffix}.json"

    @property
    def manifest_filename(self) -> str:
        return f"manifest{self.suffix}.json"

    def sessions_raw_filename(self, split: Split) -> str:
        return f"sessions_raw_{split}{self.suffix}.parquet"

    def prod_results_filename(self, split: Split) -> str:
        return f"prod_results_{split}{self.suffix}.parquet"

    def model_results_filename(self, split: Split, name: str) -> str:
        return f"model_results_{name}_{split}{self.suffix}.parquet"

    def model_rows_staging_filename(self, split: Split, name: str) -> str:
        return f"_staging_model_rows_{name}_{split}{self.suffix}.parquet"

    def inference_manifest_filename(self, split: Split, name: str) -> str:
        return f"inference_manifest_{name}_{split}{self.suffix}.json"

    def product_metadata_filename(self, split: Split, name: str) -> str:
        return f"product_metadata_{name}_{split}{self.suffix}.parquet"

    def trigger_metrics_filename(self, split: Split, name: str) -> str:
        return f"trigger_metrics_{name}_{split}{self.suffix}.parquet"

    def stats_filename(self, split: Split, name: str) -> str:
        return f"stats_{name}_{split}{self.suffix}.json"


class GraphPipelineParameters(BaseModel):
    """Configuration du pipeline de construction du graphe de co-vue.

    Reprend les constantes que graph_pipeline_prototype portait en cellules : fenêtre de dates,
    timeout de session, profondeur du classement prod, destination des artefacts. Les noms de
    tables et l'identifiant du modèle d'embedding restent des constantes de sources.py, qui sont
    l'identité de la source et non de la configuration.

    Les coupures sont exprimées en jours avant end_date pour qu'un décalage de la fenêtre
    conserve la même taille de val et de test.
    """

    model_config = ConfigDict(extra="forbid")

    seed: int | None = 42
    customer_shortname: str
    start_date: date
    end_date: date
    cutoff_train_days_before_end: int = Field(default=30, gt=0)
    cutoff_val_days_before_end: int = Field(default=15, gt=0)
    session_timeout: int = Field(default=1800, gt=0)
    symmetric_edges: bool = True
    top_k: int = Field(default=12, gt=0)
    prod_table_splits: list[Split] = Field(default_factory=lambda: ["val"])
    max_nodes_without_embedding_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    expected_num_nodes: int | None = Field(default=None, gt=0)
    artifacts: ArtifactParameters = Field(default_factory=ArtifactParameters)

    @property
    def cutoff_train(self) -> date:
        return self.end_date - timedelta(days=self.cutoff_train_days_before_end)

    @property
    def cutoff_val(self) -> date:
        return self.end_date - timedelta(days=self.cutoff_val_days_before_end)

    @model_validator(mode="after")
    def check_cutoff_ordering(self) -> "GraphPipelineParameters":
        """Impose start_date < cutoff_train < cutoff_val < end_date.

        Un ordre cassé ne lève rien à l'exécution, il vide silencieusement un split.
        """
        bounds = [self.start_date, self.cutoff_train, self.cutoff_val, self.end_date]
        if not all(earlier < later for earlier, later in pairwise(bounds)):
            raise ValueError(
                "Ordre des dates invalide : start_date < cutoff_train < cutoff_val < end_date attendu, "
                f"obtenu {self.start_date} / {self.cutoff_train} / {self.cutoff_val} / {self.end_date}"
            )
        return self

    @model_validator(mode="after")
    def check_prod_table_splits(self) -> "GraphPipelineParameters":
        """Interdit les doublons, qui feraient écrire deux fois le même parquet."""
        if len(set(self.prod_table_splits)) != len(self.prod_table_splits):
            raise ValueError(f"Doublon dans prod_table_splits : {self.prod_table_splits}")
        return self
