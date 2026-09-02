import torch
from torch_geometric.data import Data

from fast_gnn_benchmark.models.backbones import load_backbone
from fast_gnn_benchmark.models.base_model import BaseGNN
from fast_gnn_benchmark.models.embedder import load_embedder
from fast_gnn_benchmark.models.link_prediction_heads import (
    CosineSimilarityClassifier,
    Hadamard_MLPPredictor,
    MLP_CosinePredictor,
)
from fast_gnn_benchmark.metrics.base_metrics import MetricsCollection
from fast_gnn_benchmark.schemas.model import (
    ArchitectureParametersChoices,
    EmbedderParameters,
    LinkPredictionModelParameters,
    LinkPredictorParameters,
    LinkPredictorType,
)


class LinkPredictorBase(torch.nn.Module):
    def __init__(
        self,
        embedder_parameters: EmbedderParameters,
        architecture_parameters: ArchitectureParametersChoices,
        link_predictor_parameters: LinkPredictorParameters,
    ):
        super().__init__()
        self.embedder_parameters = embedder_parameters
        self.architecture_parameters = architecture_parameters
        self.link_predictor_parameters = link_predictor_parameters
        self.embedder = load_embedder(embedder_parameters)
        self.backbone = load_backbone(architecture_parameters)
        self.classifier = self.load_classifier()

    def load_classifier(self) -> torch.nn.Module:
        match self.link_predictor_parameters.link_predictor_type:
            case LinkPredictorType.COSINE_SIMILARITY:
                return CosineSimilarityClassifier()

            case LinkPredictorType.MLP_COSINE:
                return MLP_CosinePredictor(
                    hidden_dim=self.architecture_parameters.output_dim,
                    **self.link_predictor_parameters.parameters,
                )

            case LinkPredictorType.HADAMARD_MLP:
                return Hadamard_MLPPredictor(
                    hidden_dim=self.architecture_parameters.output_dim,
                    **self.link_predictor_parameters.parameters,
                )

            case _:
                raise ValueError(f"Invalid classifier type: {self.link_predictor_parameters.link_predictor_type}")

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edges = data.edge_index
        x = self.embedder(x)
        x = self.backbone(x, edges)

        return self.classifier(x, x, data.target_edges)


class LinkPredictionModel(BaseGNN[LinkPredictionModelParameters]):
    def __init__(self, model_parameters: LinkPredictionModelParameters):
        super().__init__(model_parameters)
        self.val_grouped_metrics = self._load_grouped_metrics(prefix="val/")
        self.test_grouped_metrics = self._load_grouped_metrics(prefix="test/")
        self._val_embeddings = None
        self._test_embeddings = None

    def _load_grouped_metrics(self, prefix: str) -> MetricsCollection | None:
        if not self.model_parameters.grouped_metrics:
            return None
        metrics = {m.display_name: m.get() for m in self.model_parameters.grouped_metrics}
        return MetricsCollection(metrics, prefix=prefix)

    def load_model(self) -> torch.nn.Module:
        return LinkPredictorBase(
            self.model_parameters.embedder_parameters,
            self.model_parameters.architecture_parameters,
            self.model_parameters.link_predictor_parameters,
        )

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        if self.val_grouped_metrics is not None:
            self.val_grouped_metrics.reset()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        if self.test_grouped_metrics is not None:
            self.test_grouped_metrics.reset()

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        pred = self.model(batch)
        loss = self.loss(pred, batch.y)
        batch_metrics = self.train_metrics(pred, batch.y)
        self.log_dict(
            {"train/loss": loss, **batch_metrics},
            on_step=True,
            on_epoch=True,
            batch_size=batch.y.shape[0],  # type: ignore
            prog_bar=False,
        )

        return loss

    def on_validation_epoch_end(self) -> None:
        self._val_embeddings = None
        if self.val_grouped_metrics is not None:
            self.log_dict(self.val_grouped_metrics.compute())

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        if self._val_embeddings is None:
            x = self.model.embedder(batch.x)
            self._val_embeddings = self.model.backbone(x, batch.edge_index)

        pred = self.model.classifier(self._val_embeddings, self._val_embeddings, batch.target_edges)
        loss = self.loss(pred, batch.y)
        batch_metrics = self.val_metrics(pred, batch.y)
        if self.val_grouped_metrics is not None and batch.group_ids is not None:
            self.val_grouped_metrics(pred, batch.y, batch.group_ids)
        self.log_dict({"val/loss": loss, **batch_metrics}, on_epoch=True, batch_size=batch.y.shape[0], prog_bar=False)  # type: ignore

        return loss

    def on_test_epoch_end(self) -> None:
        self._test_embeddings = None
        if self.test_grouped_metrics is not None:
            self.log_dict(self.test_grouped_metrics.compute())

    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        if self._test_embeddings is None:
            x = self.model.embedder(batch.x)
            self._test_embeddings = self.model.backbone(x, batch.edge_index)

        pred = self.model.classifier(self._test_embeddings, self._test_embeddings, batch.target_edges)
        loss = self.loss(pred, batch.y)
        batch_metrics = self.test_metrics(pred, batch.y)
        if self.test_grouped_metrics is not None and batch.group_ids is not None:
            self.test_grouped_metrics(pred, batch.y, batch.group_ids)
        self.log_dict({"test/loss": loss, **batch_metrics}, on_epoch=True, batch_size=batch.y.shape[0], prog_bar=False)  # type: ignore

        return loss
