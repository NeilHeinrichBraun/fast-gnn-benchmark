from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fast_gnn_benchmark.schemas.pipeline_model import ArtifactParameters, Split

RetrievalStrategy = Literal["exhaustive", "faiss_flat", "faiss_ivf"]
FAISS_STRATEGIES: tuple[RetrievalStrategy, ...] = ("faiss_flat", "faiss_ivf")


class CheckpointParameters(BaseModel):
    """Checkpoint à charger.

    path vaut "latest" pour prendre le .ckpt le plus récent de dir (save_top_k=1 : c'est le
    meilleur epoch), ou un chemin explicite pour figer le modèle évalué.
    """

    model_config = ConfigDict(extra="forbid")

    dir: str
    path: str = "latest"


class RetrievalParameters(BaseModel):
    """Stratégie de retrieval du top-k parmi les candidats de la catégorie du trigger.

    exhaustive score chaque paire avec le classifieur : seule stratégie compatible avec
    hadamard_mlp, dont le score n'est pas décomposable. Les stratégies faiss indexent les vecteurs
    de head.project() et exigent une tête cosine_similarity ou mlp_cosine.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: RetrievalStrategy
    batch_size: int = Field(default=8192, gt=0)
    min_pool_for_ann: int = Field(default=2048, gt=0)
    nprobe: int = Field(default=8, gt=0)
    ivf_seed: int = 42
    reference_sample_size: int = Field(default=500, ge=0)
    reference_atol: float = Field(default=1e-5, gt=0)
    min_recall_vs_flat: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_recall_threshold(self) -> "RetrievalParameters":
        """Le recall contre l'index exact n'a de sens que pour l'IVF."""
        if self.min_recall_vs_flat is not None and self.strategy != "faiss_ivf":
            raise ValueError(f"min_recall_vs_flat ne s'applique qu'à faiss_ivf, pas à {self.strategy}")
        return self


class InferenceParameters(BaseModel):
    """Configuration du job d'inférence.

    Reprend les constantes de model_test_prototype_clean et model_test_prototype_ann_clean.
    artifacts est le même bloc que dans la config du pipeline : les entrées (graphe, mappings,
    tables prod) sont relues avec le même suffixe que celui qui les a écrites.

    output_name distingue les sorties de deux modèles ou deux stratégies sur le même split : sans
    lui, flat et IVF s'écrasaient au même chemin dans le notebook ANN.
    """

    model_config = ConfigDict(extra="forbid")

    customer_shortname: str
    split: Split = "val"
    top_k: int = Field(default=12, gt=0)
    output_name: str = Field(pattern=r"^[a-z0-9_]+$")
    checkpoint: CheckpointParameters
    retrieval: RetrievalParameters
    artifacts: ArtifactParameters = Field(default_factory=ArtifactParameters)
