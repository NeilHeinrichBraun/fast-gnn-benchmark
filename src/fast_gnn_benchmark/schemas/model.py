from collections.abc import Iterable
from enum import Enum
from typing import Annotated, Any, Literal

import lightning as L
import torch
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pydantic import BaseModel, Field, field_validator

from fast_gnn_benchmark.metrics.base_metrics import (
    MRR,
    MRR_trigger,
    HitRate_trigger,
    BinaryAccuracy_trigger,
    Precision_trigger,
    Recall_trigger,
    BinaryAccuracy,
    BinaryDistribution,
    HitRate,
    OptimizedF1Score,
    OptimizedMetric,
    OptimizedMultiClassAccuracy,
    OptimizedPrecision,
    OptimizedRecall,
)
from fast_gnn_benchmark.schemas.data_models import DataParameters

# -------------------- Loss --------------------


class LossType(Enum):
    CROSS_ENTROPY = "cross_entropy"
    BCE_WITH_LOGITS_LOSS = "bce_with_logits_loss"


class LossParameters(BaseModel):
    loss_type: LossType
    parameters: dict[str, Any] = Field(default_factory=dict)

    def get(self) -> torch.nn.Module:
        match self.loss_type:
            case LossType.CROSS_ENTROPY:
                return torch.nn.CrossEntropyLoss(**self.parameters)
            case LossType.BCE_WITH_LOGITS_LOSS:
                return torch.nn.BCEWithLogitsLoss(**self.parameters)
            case _:
                raise ValueError(f"Invalid loss type: {self.loss_type}")


# -------------------- Metrics --------------------


class MetricType(Enum):
    ACCURACY = "accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    ROC_AUC = "roc_auc"
    BINARY_DISTRIBUTION = "binary_distribution"

    OPTIMIZED_ACCURACY = "optimized_accuracy"
    OPTIMIZED_F1 = "optimized_f1"
    OPTIMIZED_PRECISION = "optimized_precision"
    OPTIMIZED_RECALL = "optimized_recall"

    HIT_RATE = "hit_rate"
    MEAN_RECIPROCAL_RANK = "mean_reciprocal_rank"
    BINARY_ACCURACY = "binary_accuracy"
    MRR_TRIGGER = "mrr_trigger"
    HIT_RATE_TRIGGER = "hit_rate_trigger"
    BINARY_ACCURACY_TRIGGER = "binary_accuracy_trigger"
    PRECISION_TRIGGER = "precision_trigger"
    RECALL_TRIGGER = "recall_trigger"


class MetricParameters(BaseModel):
    metric_type: MetricType
    display_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    def get(self) -> OptimizedMetric | torchmetrics.Metric:  # noqa: PLR0911
        match self.metric_type:
            case MetricType.BINARY_DISTRIBUTION:
                return BinaryDistribution(**self.parameters)
            case MetricType.ACCURACY:
                if "task" not in self.parameters:
                    raise ValueError(
                        "Accuracy metric must have a task parameter among ['binary', 'multiclass', 'multilabel']"
                    )
                return torchmetrics.Accuracy(**self.parameters)
            case MetricType.F1:
                if "task" not in self.parameters:
                    raise ValueError(
                        "F1 metric must have a task parameter among ['binary', 'multiclass', 'multilabel']"
                    )
                return torchmetrics.F1Score(**self.parameters)
            case MetricType.PRECISION:
                if "task" not in self.parameters:
                    raise ValueError(
                        "Precision metric must have a task parameter among ['binary', 'multiclass', 'multilabel']"
                    )
                return torchmetrics.Precision(**self.parameters)
            case MetricType.RECALL:
                if "task" not in self.parameters:
                    raise ValueError(
                        "Recall metric must have a task parameter among ['binary', 'multiclass', 'multilabel']"
                    )
                return torchmetrics.Recall(**self.parameters)
            case MetricType.ROC_AUC:
                if "task" not in self.parameters:
                    raise ValueError(
                        "ROC AUC metric must have a task parameter among ['binary', 'multiclass', 'multilabel']"
                    )
                return torchmetrics.AUROC(**self.parameters)

            case MetricType.OPTIMIZED_ACCURACY:
                return OptimizedMultiClassAccuracy(**self.parameters)
            case MetricType.OPTIMIZED_F1:
                return OptimizedF1Score(**self.parameters)
            case MetricType.OPTIMIZED_PRECISION:
                return OptimizedPrecision(**self.parameters)
            case MetricType.OPTIMIZED_RECALL:
                return OptimizedRecall(**self.parameters)

            case MetricType.HIT_RATE:
                if "k" not in self.parameters:
                    raise ValueError("Hit Rate metric must have a 'k' parameter")
                return HitRate(**self.parameters)

            case MetricType.MEAN_RECIPROCAL_RANK:
                return MRR(**self.parameters)
            case MetricType.BINARY_ACCURACY:
                return BinaryAccuracy(**self.parameters)
            case MetricType.MRR_TRIGGER:
                return MRR_trigger(**self.parameters)
            case MetricType.HIT_RATE_TRIGGER:
                if "k" not in self.parameters:
                    raise ValueError("HitRate_trigger metric must have a 'k' parameter")
                return HitRate_trigger(**self.parameters)
            case MetricType.BINARY_ACCURACY_TRIGGER:
                return BinaryAccuracy_trigger(**self.parameters)
            case MetricType.PRECISION_TRIGGER:
                return Precision_trigger(**self.parameters)
            case MetricType.RECALL_TRIGGER:
                return Recall_trigger(**self.parameters)
            case _:
                raise ValueError(f"Invalid metric type: {self.metric_type}")


# -------------------- Optimizer --------------------


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"


class OptimizerParameters(BaseModel):
    optimizer_type: OptimizerType
    parameters: dict[str, Any]

    def get(self, nn_parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        match self.optimizer_type:
            case OptimizerType.ADAM:
                return torch.optim.Adam(nn_parameters, **self.parameters)
            case OptimizerType.SGD:
                return torch.optim.SGD(nn_parameters, **self.parameters)
            case OptimizerType.ADAMW:
                return torch.optim.AdamW(nn_parameters, **self.parameters)
            case _:
                raise ValueError(f"Invalid optimizer type: {self.optimizer_type}")


class SchedulerType(Enum):
    STEP = "step"
    COSINE = "cosine"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class SchedulerParameters(BaseModel):
    scheduler_type: SchedulerType
    parameters: dict[str, Any] = Field(default_factory=dict)
    monitor: str | None = None
    interval: Literal["epoch", "step"] = "epoch"
    frequency: int = 1

    def get(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau:
        match self.scheduler_type:
            case SchedulerType.STEP:
                return torch.optim.lr_scheduler.StepLR(optimizer, **self.parameters)
            case SchedulerType.COSINE:
                return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.parameters)
            case SchedulerType.REDUCE_ON_PLATEAU:
                return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.parameters)
            case _:
                raise ValueError(f"Invalid scheduler type: {self.scheduler_type}")



# -------------------- Backbone --------------------


class ArchitectureType(Enum):
    GCN = "gcn"
    SAGE = "sage"
    GAT = "gat"
    SGC = "sgc"
    SGC2 = "sgc2"  # SGC with Dropout, Layer Norm and Skip Connections
    GCNII = "gcnii"
    MLP = "mlp"
    MLP_ADJACENCY = "mlp_adjacency"
    PMLP = "pmlp"
    SGFORMER = "sgformer"
    POLYNORMER = "polynormer"
    DGCNN = "dgcnn"


class ArchitectureParameters(BaseModel):
    input_dim: int
    hidden_dim: int
    output_dim: int


class GNNParameters(ArchitectureParameters):
    architecture_type: Literal[
        ArchitectureType.GCN,
        ArchitectureType.SAGE,
        ArchitectureType.GAT,
        ArchitectureType.SGC2,
    ]
    num_layers: int
    dropout: float = 0.0
    use_input_projection: bool = False
    use_output_projection: bool = False
    use_residual: bool = False
    use_layer_norm: bool = False
    use_batch_norm: bool = False
    conv_parameters: dict[str, Any] = Field(default_factory=dict)


class SGCParameters(ArchitectureParameters):
    architecture_type: Literal[ArchitectureType.SGC] = ArchitectureType.SGC
    num_layers: int


class GCNIIParameters(ArchitectureParameters):
    architecture_type: Literal[ArchitectureType.GCNII] = ArchitectureType.GCNII
    num_layers: int
    alpha: float
    theta: float
    dropout: float


class MLPParameters(ArchitectureParameters):
    architecture_type: Literal[ArchitectureType.MLP] = ArchitectureType.MLP
    num_layers: int
    dropout: float = 0.0
    use_layer_norm: bool = False
    use_residual: bool = False


class MLPAdjacencyParameters(ArchitectureParameters):
    architecture_type: Literal[ArchitectureType.MLP_ADJACENCY] = ArchitectureType.MLP_ADJACENCY
    num_layers: int
    use_layer_norm: bool = False
    use_residual: bool = False
    dropout: float = 0.0


class PMLPParameters(ArchitectureParameters):
    architecture_type: Literal[ArchitectureType.PMLP] = ArchitectureType.PMLP
    num_layers: int
    dropout: float = 0.0


class SGFormerParameters(ArchitectureParameters):
    architecture_type: Literal[ArchitectureType.SGFORMER] = ArchitectureType.SGFORMER
    num_layers: int
    alpha: float
    dropout: float
    num_heads: int
    use_layer_norm: bool
    use_residual: bool
    use_graph: bool
    use_weight: bool
    graph_weight: float

    gnn_parameters: GNNParameters

    @field_validator("gnn_parameters", mode="before")
    @classmethod
    def convert_gnn_architecture_type(cls, v):
        if isinstance(v, dict) and "architecture_type" in v:
            gnn_arch_type = v["architecture_type"]
            if isinstance(gnn_arch_type, str):
                try:
                    v = v.copy()  # Don't modify the original
                    v["architecture_type"] = ArchitectureType(gnn_arch_type)
                except ValueError:
                    raise ValueError(
                        f"Invalid gnn_architecture_type: {gnn_arch_type}. Must be one of: {[e.value for e in ArchitectureType]}"
                    ) from None
        return v


class PolyNormerParameters(ArchitectureParameters):
    architecture_type: Literal[ArchitectureType.POLYNORMER] = ArchitectureType.POLYNORMER
    local_layers: int
    global_layers: int
    in_dropout: float
    local_dropout: float
    global_dropout: float
    num_heads: int
    beta: float = 0.9
    pre_norm: bool = False
    qk_shared: bool = False

class DGCNNParameters(ArchitectureParameters):
    architecture_type: Literal[ArchitectureType.DGCNN] = ArchitectureType.DGCNN
    num_layers: int
    k: int = 30
    inner_gnn_type: Literal[ArchitectureType.GCN, ArchitectureType.SAGE, ArchitectureType.GAT]
    conv1d_channels: list[int]
    conv1d_kernel_size: int = 5
    mlp_hidden_dim: int = 128
    dropout: float = 0.5

    @field_validator("inner_gnn_type", mode="before")
    @classmethod
    def convert_inner_gnn_type(cls, v):
        if isinstance(v, str):
            try:
                return ArchitectureType(v)
            except ValueError:
                raise ValueError(
                    f"Invalid inner_gnn_type: {v}. Must be one of: ['gcn', 'sage', 'gat']"
                ) from None
        return v


# -------------------- Base model --------------------

ArchitectureParametersChoices = Annotated[
    GNNParameters
    | MLPParameters
    | MLPAdjacencyParameters
    | SGFormerParameters
    | PolyNormerParameters
    | SGCParameters
    | PMLPParameters
    | GCNIIParameters
    | DGCNNParameters,
    Field(discriminator="architecture_type"),
]


class EmbedderParameters(BaseModel):
    use_embedding: bool
    embedding_dim: int
    embedding_only: bool
    initializer: Literal["orthogonal", "uniform", "ones", "normal"]
    num_nodes: int


class BaseModelParameters(BaseModel):
    architecture_parameters: ArchitectureParametersChoices
    loss: LossParameters
    metrics: list[MetricParameters]
    optimizer: OptimizerParameters
    scheduler: SchedulerParameters | None = None

    @field_validator("architecture_parameters", mode="before")
    @classmethod
    def convert_architecture_type(cls, v):
        if isinstance(v, dict) and "architecture_type" in v:
            arch_type = v["architecture_type"]
            if isinstance(arch_type, str):
                try:
                    v = v.copy()  # Don't modify the original
                    v["architecture_type"] = ArchitectureType(arch_type)
                except ValueError:
                    raise ValueError(
                        f"Invalid architecture_type: {arch_type}. Must be one of: {[e.value for e in ArchitectureType]}"
                    ) from None
        return v


# -------------------- Task-specific models --------------------


class NodeClassificationModelParameters(BaseModelParameters):
    task_type: Literal["node_classification"] = "node_classification"


class LinkPredictorType(Enum):
    COSINE_SIMILARITY = "cosine_similarity"
    HADAMARD_MLP = "hadamard_mlp"


class LinkPredictorParameters(BaseModel):
    link_predictor_type: LinkPredictorType | None
    parameters: dict[str, Any]


class LinkPredictionModelParameters(BaseModelParameters):
    task_type: Literal["link_prediction"] = "link_prediction"
    task_subtype: Literal["whole_graph", "sub_graph"] = "whole_graph"
    link_predictor_parameters: LinkPredictorParameters
    embedder_parameters: EmbedderParameters = Field(
        default_factory=lambda: EmbedderParameters(
            use_embedding=False, embedding_dim=0, embedding_only=False, initializer="orthogonal", num_nodes=0
        )
    )
    grouped_metrics: list[MetricParameters] = Field(default_factory=list)


# -------------------- Trainer parameters --------------------


class CallbackType(Enum):
    EARLY_STOPPING = "early_stopping"
    MODEL_CHECKPOINT = "model_checkpoint"


class CallbackParameters(BaseModel):
    callback_type: CallbackType
    parameters: dict[str, Any] = Field(default_factory=dict)

    def get(self) -> L.Callback:
        match self.callback_type:
            case CallbackType.EARLY_STOPPING:
                for key in ["monitor", "patience", "mode"]:
                    if key not in self.parameters:
                        raise ValueError(f"Early stopping callback must have a {key} parameter")
                return EarlyStopping(**self.parameters)
            case CallbackType.MODEL_CHECKPOINT:
                for key in ["monitor", "mode"]:
                    if key not in self.parameters:
                        raise ValueError(f"Model checkpoint callback must have a {key} parameter")
                return ModelCheckpoint(**self.parameters)
            case _:
                raise ValueError(f"Invalid callback type: {self.callback_type}")


class WandbLoggerParameters(BaseModel):
    entity: str
    project: str
    reinit: bool | str = False
    id: str | None = None  # wandb run id
    name: str | None = None  # wandb run name
    tags: list[str] = Field(default_factory=list)

    def get(self) -> WandbLogger:
        logger = WandbLogger(**self.model_dump())

        self.id = logger.experiment.id
        self.name = logger.experiment.name

        return logger


class CompilationParameters(BaseModel):
    use_compiled_torch: bool = True
    full_graph: bool = False


class TrainerParameters(BaseModel):
    seed: int | None = None
    data_parameters: DataParameters
    model_parameters: Annotated[
        NodeClassificationModelParameters | LinkPredictionModelParameters,
        Field(discriminator="task_type"),
    ]
    callbacks: list[CallbackParameters]
    wandb_logger_parameters: WandbLoggerParameters | None = None
    compilation_parameters: CompilationParameters = Field(default_factory=CompilationParameters)
    trainer_config: dict[str, Any] = Field(default_factory=dict)
