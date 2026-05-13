import torch
from torch_geometric.data import Data

from fast_gnn_benchmark.models.backbones import load_backbone
from fast_gnn_benchmark.models.base_model import BaseGNN
from fast_gnn_benchmark.schemas.model import LinkPredictionModelParameters


class SEALLinkPredictionModel(BaseGNN[LinkPredictionModelParameters]):
    def __init__(self, model_parameters: LinkPredictionModelParameters):
        super().__init__(model_parameters)

    def load_model(self) -> torch.nn.Module:
        return load_backbone(self.model_parameters.architecture_parameters)

    def _forward(self, batch: Data) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.model(batch.x, batch.edge_index, batch.batch).view(-1)
        y = batch.y.float()
        return pred, y

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        pred, y = self._forward(batch)
        loss = self.loss(pred, y)
        batch_metrics = self.train_metrics(pred, y)
        self.log_dict(
            {"train/loss": loss, **batch_metrics},
            on_step=True,
            on_epoch=True,
            batch_size=y.shape[0],
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        pred, y = self._forward(batch)
        loss = self.loss(pred, y)
        batch_metrics = self.val_metrics(pred, y)
        self.log_dict(
            {"val/loss": loss, **batch_metrics},
            on_epoch=True,
            batch_size=y.shape[0],
            prog_bar=False,
        )
        return loss

    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        pred, y = self._forward(batch)
        loss = self.loss(pred, y)
        batch_metrics = self.test_metrics(pred, y)
        self.log_dict(
            {"test/loss": loss, **batch_metrics},
            on_epoch=True,
            batch_size=y.shape[0],
            prog_bar=False,
        )
        return loss
