import copy
from abc import ABC, abstractmethod

import torch
import torchmetrics


class BinaryDistribution(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("prediction_class", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:  # noqa: ARG002
        with torch.no_grad():
            self.prediction_class += (pred > 0).sum()
            self.total_samples += pred.shape[0]

    def compute(self) -> torch.Tensor:
        return self.prediction_class.float() / self.total_samples.float()  # type: ignore


# -------------------- Compilation-friendly Metrics for Masked data  --------------------


class OptimizedMetric(torch.nn.Module, ABC):
    @abstractmethod
    def update(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute(self) -> torch.Tensor:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.update(*args, **kwargs)


class MetricsCollection(torch.nn.Module):
    def __init__(self, metrics: dict[str, OptimizedMetric | torchmetrics.Metric], prefix: str):
        super().__init__()
        self.metrics = torch.nn.ModuleDict(metrics)
        self.prefix = prefix

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        results: dict[str, torch.Tensor] = {}
        for name, metric in self.metrics.items():
            results[f"{self.prefix}{name}"] = metric(*args, **kwargs)

        return results

    def compute(self) -> dict[str, torch.Tensor]:
        results: dict[str, torch.Tensor] = {}
        for name, metric in self.metrics.items():
            results[f"{self.prefix}{name}"] = metric.compute()  # type: ignore
        return results

    def reset(self) -> None:
        for metric in self.metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()  # type: ignore

    def clone(self, prefix: str) -> "MetricsCollection":
        new_metrics: dict[str, OptimizedMetric | torchmetrics.Metric] = {
            name: copy.deepcopy(metric)
            for name, metric in self.metrics.items()  # type: ignore
        }
        return MetricsCollection(new_metrics, prefix=prefix)

    def add_metrics(self, metrics: dict[str, OptimizedMetric | torchmetrics.Metric]) -> None:
        self.metrics.update(metrics)


class OptimizedMultiClassAccuracy(OptimizedMetric):
    def __init__(self):
        super().__init__()
        self.register_buffer("correct_predictions", torch.tensor(0.0))
        self.register_buffer("total_samples", torch.tensor(0.0))

        self.reset()

    @staticmethod
    def get_accuracy(correct_predictions: torch.Tensor, total_samples: torch.Tensor) -> torch.Tensor:
        return correct_predictions / total_samples.clamp(min=1)

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        with torch.no_grad():
            pred = pred.argmax(dim=1)
            batch_correct_predictions = ((pred == target) * mask).sum()
            batch_total_samples = mask.sum().clamp(min=1)

            self.correct_predictions += batch_correct_predictions
            self.total_samples += batch_total_samples

            return self.get_accuracy(batch_correct_predictions, batch_total_samples)

    def compute(self) -> torch.Tensor:
        return self.get_accuracy(self.correct_predictions, self.total_samples)

    def reset(self) -> None:
        self.correct_predictions.zero_()  # type: ignore
        self.total_samples.zero_()  # type: ignore


class OptimizedStatScores(OptimizedMetric):
    def __init__(self):
        super().__init__()
        self.register_buffer("tp", torch.tensor(0.0))
        self.register_buffer("fp", torch.tensor(0.0))
        self.register_buffer("tn", torch.tensor(0.0))
        self.register_buffer("fn", torch.tensor(0.0))
        self.reset()

    def update(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            pred = pred.argmax(dim=1)
            batch_tp = ((pred == target) * mask).sum()
            batch_fp = ((pred != target) * mask).sum()
            batch_tn = ((pred == target) * mask).sum()
            batch_fn = ((pred != target) * mask).sum()

            self.tp += batch_tp
            self.fp += batch_fp
            self.tn += batch_tn
            self.fn += batch_fn

        return batch_tp, batch_fp, batch_tn, batch_fn

    def reset(self) -> None:
        self.tp.zero_()  # type: ignore
        self.fp.zero_()  # type: ignore
        self.tn.zero_()  # type: ignore
        self.fn.zero_()  # type: ignore


class OptimizedF1Score(OptimizedStatScores):
    @staticmethod
    def get_f1_score(batch_tp: torch.Tensor, batch_fp: torch.Tensor, batch_fn: torch.Tensor) -> torch.Tensor:
        precision = batch_tp / (batch_tp + batch_fp)
        recall = batch_tp / (batch_tp + batch_fn)
        return 2 * (precision * recall) / (precision + recall)

    def update(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_tp, batch_fp, _, batch_fn = super().update(pred, target, mask)
        return self.get_f1_score(batch_tp, batch_fp, batch_fn)

    def compute(self) -> torch.Tensor:
        return self.get_f1_score(self.tp, self.fp, self.fn)  # type: ignore


class OptimizedPrecision(OptimizedStatScores):
    @staticmethod
    def get_precision(batch_tp: torch.Tensor, batch_fp: torch.Tensor) -> torch.Tensor:
        return batch_tp / (batch_tp + batch_fp)

    def update(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_tp, batch_fp, _, _ = super().update(pred, target, mask)
        return self.get_precision(batch_tp, batch_fp)

    def compute(self) -> torch.Tensor:
        return self.get_precision(self.tp, self.fp)


class OptimizedRecall(OptimizedStatScores):
    @staticmethod
    def get_recall(batch_tp: torch.Tensor, batch_fn: torch.Tensor) -> torch.Tensor:
        return batch_tp / (batch_tp + batch_fn)

    def update(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_tp, _, _, batch_fn = super().update(pred, target, mask)
        return self.get_recall(batch_tp, batch_fn)

    def compute(self) -> torch.Tensor:
        return self.get_recall(self.tp, self.fn)


# -------------------- Link prediction metrics --------------------


class HitRate(OptimizedMetric):
    def __init__(self, k: int) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        self.k = k

        # persistent=False so these accumulating buffers are not saved in checkpoints
        self.register_buffer("pos_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("neg_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        self.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        pred: logits/scores for each sample
        target: tensor containing 0.0 or 1.0 labels (same first-dim length as pred)
        """
        with torch.no_grad():
            # target could be float/bool/int; interpret >0.5 as positive
            pos_mask = target > 0.5  # noqa: PLR2004
            neg_mask = ~pos_mask

            pos_batch = pred[pos_mask]
            neg_batch = pred[neg_mask]

            # Append to buffers
            if pos_batch.numel() > 0:
                self.pos_scores = torch.cat([self.pos_scores.cpu(), pos_batch.detach().cpu()])
            if neg_batch.numel() > 0:
                self.neg_scores = torch.cat([self.neg_scores.cpu(), neg_batch.detach().cpu()])

            # Return current hit rate computed on this batch
            return self._hits_at_k(pos_batch, neg_batch, self.k)

    @staticmethod
    def _hits_at_k(y_pred_pos: torch.Tensor, y_pred_neg: torch.Tensor, k: int) -> torch.Tensor:
        """
        Vectorized version of your eval_hits, returning a scalar tensor.
        """
        # If no positive samples, define as 0 to avoid NaNs (you can choose another convention)
        if y_pred_pos.numel() == 0:
            return torch.tensor(0.0, device=y_pred_neg.device if y_pred_neg.is_cuda else y_pred_pos.device)

        # If not enough negatives, hits@k is 1.0 per your reference function
        if y_pred_neg.numel() < k:
            return torch.tensor(1.0, device=y_pred_pos.device)

        kth_score_in_neg = torch.topk(y_pred_neg, k).values[-1]
        return (y_pred_pos > kth_score_in_neg).float().mean()

    def compute(self) -> torch.Tensor:
        with torch.no_grad():
            return self._hits_at_k(self.pos_scores, self.neg_scores, self.k)

    def reset(self) -> None:
        # "Empty" the buffers while keeping device/dtype consistent
        self.pos_scores = self.pos_scores.new_empty((0,))
        self.neg_scores = self.neg_scores.new_empty((0,))


class MRR(OptimizedMetric):
    """
    Accumulates positive and negative scores across updates, then computes
    mean reciprocal rank using the same logic as eval_mrr.
    """

    def __init__(self) -> None:
        super().__init__()
        # persistent=False so these accumulating buffers are not saved in checkpoints
        self.register_buffer("pos_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("neg_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        self.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        pred: scores/logits for each sample (1D tensor)
        target: tensor containing 0.0 or 1.0 labels (same length as pred)
        """
        with torch.no_grad():
            pos_mask = target > 0.5  # noqa: PLR2004
            neg_mask = ~pos_mask

            pos_batch = pred[pos_mask].detach()
            neg_batch = pred[neg_mask].detach()

            if pos_batch.numel() > 0:
                self.pos_scores = torch.cat([self.pos_scores, pos_batch])
            if neg_batch.numel() > 0:
                self.neg_scores = torch.cat([self.neg_scores, neg_batch])

            # Return batch MRR (computed on this batch)
            return self._mrr(pos_batch, neg_batch)

    @staticmethod
    def _mrr(y_pred_pos: torch.Tensor, y_pred_neg: torch.Tensor) -> torch.Tensor:
        """
        Port of eval_mrr, but returns a scalar tensor.
        Assumes y_pred_pos are the scores for positive edges and y_pred_neg for negatives.
        """
        # Match "no positives" edge case handling similar to HitRate (avoid NaNs)
        if y_pred_pos.numel() == 0:
            device = y_pred_neg.device if y_pred_neg.is_cuda else y_pred_pos.device
            return torch.tensor(0.0, device=device)

        # eval_mrr assumes y_pred_neg is 2D (num_pos, num_neg) so it can rank per positive.
        # With the same buffering scheme as HitRate (a flat list of negatives),
        # we treat *all* accumulated negatives as the comparison set for each positive.
        y_pred_pos_2d = y_pred_pos.view(-1, 1)  # [P, 1]
        y_pred_neg_2d = y_pred_neg.view(1, -1)  # [1, N] -> broadcast to [P, N]

        optimistic_rank = (y_pred_neg_2d > y_pred_pos_2d).sum(dim=1)
        pessimistic_rank = (y_pred_neg_2d >= y_pred_pos_2d).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1.0
        mrr_list = 1.0 / ranking_list.to(torch.float32)

        return mrr_list.mean()

    def compute(self) -> torch.Tensor:
        with torch.no_grad():
            return self._mrr(self.pos_scores, self.neg_scores)

    def reset(self) -> None:
        self.pos_scores = self.pos_scores.new_empty((0,))
        self.neg_scores = self.neg_scores.new_empty((0,))


class BinaryAccuracy_trigger(OptimizedMetric):
    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold
        self.register_buffer("pos_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("neg_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("pos_group_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("neg_group_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        with torch.no_grad():
            pos_mask = target > 0.5  # noqa: PLR2004
            neg_mask = ~pos_mask

            pos_batch = pred[pos_mask].detach()
            neg_batch = pred[neg_mask].detach()
            pos_group_ids_batch = group_ids[pos_mask].detach()
            neg_group_ids_batch = group_ids[neg_mask].detach()

            if pos_batch.numel() > 0:
                self.pos_scores = torch.cat([self.pos_scores, pos_batch])
                self.pos_group_ids = torch.cat([self.pos_group_ids, pos_group_ids_batch])
            if neg_batch.numel() > 0:
                self.neg_scores = torch.cat([self.neg_scores, neg_batch])
                self.neg_group_ids = torch.cat([self.neg_group_ids, neg_group_ids_batch])

            return self._accuracy(pos_batch, neg_batch, pos_group_ids_batch, neg_group_ids_batch)

    def _accuracy(
        self,
        y_pred_pos: torch.Tensor,
        y_pred_neg: torch.Tensor,
        pos_group_ids: torch.Tensor,
        neg_group_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Threshold-based accuracy (TP + TN) / total, computed per trigger then averaged over triggers."""
        device = y_pred_pos.device if y_pred_pos.numel() > 0 else y_pred_neg.device
        all_gids = torch.cat([pos_group_ids, neg_group_ids])
        if all_gids.numel() == 0:
            return torch.tensor(0.0, device=device)

        _, inverse = all_gids.unique(return_inverse=True)
        pos_compact = inverse[: pos_group_ids.shape[0]]
        neg_compact = inverse[pos_group_ids.shape[0] :]
        num_groups = inverse.max().item() + 1

        tp = (y_pred_pos > self.threshold).float()   # positives predicted positive
        tn = (y_pred_neg <= self.threshold).float()  # negatives predicted negative

        correct = torch.zeros(num_groups, device=device)
        correct.scatter_add_(0, pos_compact, tp)
        correct.scatter_add_(0, neg_compact, tn)

        total = torch.zeros(num_groups, device=device)
        total.scatter_add_(0, pos_compact, torch.ones_like(tp))
        total.scatter_add_(0, neg_compact, torch.ones_like(tn))

        return (correct / total.clamp(min=1)).mean()

    def compute(self) -> torch.Tensor:
        with torch.no_grad():
            return self._accuracy(self.pos_scores, self.neg_scores, self.pos_group_ids, self.neg_group_ids)

    def reset(self) -> None:
        self.pos_scores = self.pos_scores.new_empty((0,))
        self.neg_scores = self.neg_scores.new_empty((0,))
        self.pos_group_ids = self.pos_group_ids.new_empty((0,))
        self.neg_group_ids = self.neg_group_ids.new_empty((0,))


class HitRate_trigger(OptimizedMetric):
    def __init__(self, k: int) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        self.k = k
        self.register_buffer("pos_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("neg_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("pos_group_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("neg_group_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        with torch.no_grad():
            pos_mask = target > 0.5  # noqa: PLR2004
            neg_mask = ~pos_mask

            pos_batch = pred[pos_mask].detach()
            neg_batch = pred[neg_mask].detach()
            pos_group_ids_batch = group_ids[pos_mask].detach()
            neg_group_ids_batch = group_ids[neg_mask].detach()

            if pos_batch.numel() > 0:
                self.pos_scores = torch.cat([self.pos_scores, pos_batch])
                self.pos_group_ids = torch.cat([self.pos_group_ids, pos_group_ids_batch])
            if neg_batch.numel() > 0:
                self.neg_scores = torch.cat([self.neg_scores, neg_batch])
                self.neg_group_ids = torch.cat([self.neg_group_ids, neg_group_ids_batch])

            return self._hits_at_k(pos_batch, neg_batch, self.k, pos_group_ids_batch, neg_group_ids_batch)

    @staticmethod
    def _hits_at_k(
        y_pred_pos: torch.Tensor,
        y_pred_neg: torch.Tensor,
        k: int,
        pos_group_ids: torch.Tensor,
        neg_group_ids: torch.Tensor,
    ) -> torch.Tensor:
        if y_pred_pos.numel() == 0:
            return torch.tensor(0.0, device=y_pred_pos.device)

        hits = []
        for group_id in pos_group_ids.unique():
            pos = y_pred_pos[pos_group_ids == group_id]
            neg = y_pred_neg[neg_group_ids == group_id]

            if neg.numel() < k:
                hits.append(torch.tensor(1.0, device=pos.device))
            else:
                kth_score = torch.topk(neg, k).values[-1]
                hits.append((pos > kth_score).float().mean())

        return torch.stack(hits).mean()

    def compute(self) -> torch.Tensor:
        with torch.no_grad():
            return self._hits_at_k(self.pos_scores, self.neg_scores, self.k, self.pos_group_ids, self.neg_group_ids)

    def reset(self) -> None:
        self.pos_scores = self.pos_scores.new_empty((0,))
        self.neg_scores = self.neg_scores.new_empty((0,))
        self.pos_group_ids = self.pos_group_ids.new_empty((0,))
        self.neg_group_ids = self.neg_group_ids.new_empty((0,))


class MRR_trigger(OptimizedMetric):

    def __init__(self) -> None:
        super().__init__()

        self.register_buffer("pos_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("neg_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        
        self.register_buffer("pos_group_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("neg_group_ids", torch.empty(0, dtype=torch.long), persistent=False)
        
        self.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            pos_mask = target > 0.5  
            neg_mask = ~pos_mask

            pos_batch = pred[pos_mask].detach()
            neg_batch = pred[neg_mask].detach()

            pos_group_ids_batch = group_ids[pos_mask].detach()
            neg_group_ids_batch = group_ids[neg_mask].detach()

            if pos_batch.numel() > 0:
                self.pos_scores = torch.cat([self.pos_scores, pos_batch])
                self.pos_group_ids = torch.cat([self.pos_group_ids, group_ids[pos_mask].detach()])
            
            if neg_batch.numel() > 0:
                self.neg_scores = torch.cat([self.neg_scores, neg_batch])
                self.neg_group_ids = torch.cat([self.neg_group_ids, group_ids[neg_mask].detach()])

            return self._mrr(pos_batch, neg_batch, pos_group_ids_batch, neg_group_ids_batch)

    @staticmethod
    def _mrr(
        y_pred_pos: torch.Tensor,
        y_pred_neg: torch.Tensor,
        pos_group_ids: torch.Tensor,
        neg_group_ids: torch.Tensor,
    ) -> torch.Tensor:
        
        if y_pred_pos.numel() == 0:
            return torch.tensor(0.0, device=y_pred_pos.device)
        
        device = y_pred_pos.device
        P, N = y_pred_pos.numel(), y_pred_neg.numel()

        # 1. Combined tab 
        all_scores = torch.cat([y_pred_pos, y_pred_neg])
        all_groups = torch.cat([pos_group_ids, neg_group_ids])
        is_neg = torch.cat([
            torch.zeros(P, dtype=torch.long, device=device),  # positifs -> 0
            torch.ones(N,  dtype=torch.long, device=device),  # negatifs -> 1
        ])

        # 2. Ranking per group first and then per decreasing score
        order = torch.argsort(-all_scores, stable=True)
        sort_idx = order[torch.argsort(all_groups[order], stable=True)]
        sorted_groups = all_groups[sort_idx]
        sorted_is_neg = is_neg[sort_idx]

        # 3. Define the frontier between groups 
        is_new_group = torch.cat([
            torch.tensor([True], device=device),
            sorted_groups[1:] != sorted_groups[:-1],
        ])
        group_idx = is_new_group.cumsum(0) - 1

        # 4. Negative cumsum with reset per group 
        global_cumsum = sorted_is_neg.cumsum(0)
        group_starts = torch.where(is_new_group)[0]
        cumsum_at_start = torch.cat([
            torch.zeros(1, dtype=global_cumsum.dtype, device=device),
            global_cumsum[group_starts[1:] - 1],
        ])
        within_group_neg_count = global_cumsum - cumsum_at_start[group_idx]

        # 5. Rank of the positives
        pos_mask = ~sorted_is_neg.bool()
        pos_ranks = within_group_neg_count[pos_mask].float() + 1.0
        pos_group_idx = group_idx[pos_mask]
        rr = 1.0 / pos_ranks

        # 6. Mean per group
        num_groups = group_starts.shape[0]
        rr_sum = torch.zeros(num_groups, device=device).scatter_add_(
            0, pos_group_idx, rr)
        pos_count = torch.zeros(num_groups, device=device).scatter_add_(
            0, pos_group_idx, torch.ones_like(rr))
        valid = pos_count > 0
        group_mrr = rr_sum[valid] / pos_count[valid]

        # 7. Final mean
        return group_mrr.mean()
    
    
    def compute(self) -> torch.Tensor:
        with torch.no_grad():
            return self._mrr(self.pos_scores, self.neg_scores, self.pos_group_ids, self.neg_group_ids)

    def reset(self) -> None:
        self.pos_scores = self.pos_scores.new_empty((0,))
        self.neg_scores = self.neg_scores.new_empty((0,))
        self.pos_group_ids = self.pos_group_ids.new_empty((0,))
        self.neg_group_ids = self.neg_group_ids.new_empty((0,))


class Precision_trigger(OptimizedMetric):
    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold

        self.register_buffer("pos_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("neg_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        
        self.register_buffer("pos_group_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("neg_group_ids", torch.empty(0, dtype=torch.long), persistent=False)
        
        self.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pos_mask = target > 0.5  
            neg_mask = ~pos_mask

            pos_batch = pred[pos_mask].detach()
            neg_batch = pred[neg_mask].detach()
            pos_group_ids_batch = group_ids[pos_mask].detach()
            neg_group_ids_batch = group_ids[neg_mask].detach()

            self.pos_scores = torch.cat([self.pos_scores, pos_batch])
            self.neg_scores = torch.cat([self.neg_scores, neg_batch])

            self.pos_group_ids = torch.cat([self.pos_group_ids, pos_group_ids_batch])
            self.neg_group_ids = torch.cat([self.neg_group_ids, neg_group_ids_batch])

            return self._precision(pred[pos_mask], pred[neg_mask], group_ids[pos_mask], group_ids[neg_mask])
        
    def _precision(self, pos, neg, pos_gids, neg_gids):
        device = pos.device if pos.numel() > 0 else neg.device
        all_gids_cat = torch.cat([pos_gids, neg_gids])
        if all_gids_cat.numel() == 0:
            return torch.tensor(0.0, device=device)

        _, inverse = all_gids_cat.unique(return_inverse=True)
        pos_compact = inverse[:pos_gids.shape[0]]
        neg_compact = inverse[pos_gids.shape[0]:]
        num_groups = inverse.max().item() + 1

        pos_above = (pos > self.threshold).float()
        neg_above = (neg > self.threshold).float()

        tp = torch.zeros(num_groups, device=device).scatter_add_(0, pos_compact, pos_above)
        fp = torch.zeros(num_groups, device=device).scatter_add_(0, neg_compact, neg_above)

        return (tp / (tp + fp).clamp(min=1)).mean()
    
    def compute(self):
        with torch.no_grad():
            return self._precision(self.pos_scores, self.neg_scores,
                                   self.pos_group_ids, self.neg_group_ids)
        
    def reset(self):
        self.pos_scores = self.pos_scores.new_empty((0,))
        self.neg_scores = self.neg_scores.new_empty((0,))
        self.pos_group_ids = self.pos_group_ids.new_empty((0,))
        self.neg_group_ids = self.neg_group_ids.new_empty((0,))

class Recall_trigger(OptimizedMetric):
    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold

        self.register_buffer("pos_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("pos_group_ids", torch.empty(0, dtype=torch.long), persistent=False)
        
        self.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pos_mask = target > 0.5  

            pos_batch = pred[pos_mask].detach()
            pos_group_ids_batch = group_ids[pos_mask].detach()

            self.pos_scores = torch.cat([self.pos_scores, pos_batch])
            self.pos_group_ids = torch.cat([self.pos_group_ids, pos_group_ids_batch])

            return self._recall(pos_batch, pos_group_ids_batch)
        
    def _recall(self, pos, pos_gids):
        if pos.numel() == 0:
            return torch.tensor(0.0, device=pos.device)

        _, compact = pos_gids.unique(return_inverse=True)
        num_groups = compact.max().item() + 1
        device = pos.device

        pos_above = (pos > self.threshold).float()

        tp = torch.zeros(num_groups, device=device).scatter_add_(0, compact, pos_above)
        total = torch.zeros(num_groups, device=device).scatter_add_(0, compact, torch.ones_like(pos_above))

        return (tp / total.clamp(min=1)).mean()
    
    def compute(self):
        with torch.no_grad():
            return self._recall(self.pos_scores, self.pos_group_ids)
        
    def reset(self):
        self.pos_scores = self.pos_scores.new_empty((0,))
        self.pos_group_ids = self.pos_group_ids.new_empty((0,))


class BinaryAccuracy(OptimizedMetric):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("correct_predictions", torch.tensor(0.0))
        self.register_buffer("total_samples", torch.tensor(0.0))
        self.reset()

    @staticmethod
    def get_accuracy(correct_predictions: torch.Tensor, total_samples: torch.Tensor) -> torch.Tensor:
        return correct_predictions / total_samples.clamp(min=1)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        with torch.no_grad():
            correct_predictions = (pred.sigmoid() > 0.5) == target  # noqa: PLR2004
            self.correct_predictions += correct_predictions.sum()
            self.total_samples += pred.shape[0]

            return self.get_accuracy(self.correct_predictions, self.total_samples)  # type: ignore

    def compute(self) -> torch.Tensor:
        return self.get_accuracy(self.correct_predictions, self.total_samples)

    def reset(self) -> None:
        self.correct_predictions.zero_()  # type: ignore
        self.total_samples.zero_()  # type: ignore
