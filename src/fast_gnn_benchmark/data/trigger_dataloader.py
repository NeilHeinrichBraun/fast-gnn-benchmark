from collections.abc import Iterator
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor

from fast_gnn_benchmark.data.utils import to_undirected
from fast_gnn_benchmark.schemas.dataset_models import SplitType

class TriggerLoader:
    def __init__(
            self,
            dataset: Any,
            batch_size: int,
            mask_loss_edges: bool = True,
            on_device=True,
            split_type: SplitType = SplitType.TRAIN,
            use_val_edges_as_input: bool = False,
            neg_sampling_ratio: int = 5,
    ):

        self.device = torch.device("cuda" if on_device and torch.cuda.is_available() else "cpu")

        self._use_sparse = True

        self.data = dataset.data.to(self.device)
        self.num_nodes = dataset.num_nodes
        self.batch_size = batch_size
        self.mask_loss_edges = mask_loss_edges
        self.split_type = split_type
        self.neg_sampling_ratio = neg_sampling_ratio

        self.use_val_edges_as_input = use_val_edges_as_input

        self.to_sparse_tensor = ToSparseTensor() if self._use_sparse else None

        match split_type:
            case SplitType.TRAIN:
                splits = dataset.split["train"]
                positive_edges = splits["edge"].T
                negative_edges = splits["edge_neg"].T

                self.pos_edges_train = positive_edges[1:, :].to(self.device)
                self.all_neg_edges_train = negative_edges[1:, :].to(self.device)

                if self.mask_loss_edges:
                    self.data.edge_index = self._remove_edges_from_graph(
                        self.data.edge_index, self.pos_edges_train
                    )
            
            case SplitType.VAL:
                splits = dataset.split["valid"]
                positive_edges = splits["edge"].T
                negative_edges = splits["edge_neg"].T

                exec_codes_positive = positive_edges[0,:]
                node_edges_positive = positive_edges[1:, :]

                exec_codes_negative = negative_edges[0,:]
                node_edges_negative = negative_edges[1:, :]

                exec_codes_negative, node_edges_negative = self._sample_negatives_per_group(
                    exec_codes_positive, exec_codes_negative, node_edges_negative
                )

                self.target_edges = torch.cat([node_edges_positive, node_edges_negative], dim=1).to(self.device)

                self.group_ids = torch.cat([exec_codes_positive, exec_codes_negative], dim=0).to(self.device)

                self.labels = torch.cat(
                    [torch.ones(node_edges_positive.shape[1]), torch.zeros(node_edges_negative.shape[1])], dim=0
                ).to(self.device)

            case SplitType.TEST:
                splits = dataset.split["test"]
                positive_edges = splits["edge"].T
                negative_edges = splits["edge_neg"].T

                exec_codes_positive = positive_edges[0,:]
                node_edges_positive = positive_edges[1:, :]

                exec_codes_negative = negative_edges[0,:]
                node_edges_negative = negative_edges[1:, :]

                exec_codes_negative, node_edges_negative = self._sample_negatives_per_group(
                    exec_codes_positive, exec_codes_negative, node_edges_negative
                )

                self.target_edges = torch.cat([node_edges_positive, node_edges_negative], dim=1).to(self.device)

                self.group_ids = torch.cat([exec_codes_positive, exec_codes_negative], dim=0).to(self.device)

                self.labels = torch.cat(
                    [torch.ones(node_edges_positive.shape[1]), torch.zeros(node_edges_negative.shape[1])], dim=0
                ).to(self.device)

                if use_val_edges_as_input:
                    val_edges = dataset.split["valid"]["edge"].T[1:, :].to(self.device)
                    self.data = Data(
                        x=self.data.x,
                        edge_index=torch.cat([self.data.edge_index, to_undirected(val_edges)], dim=1),
                    )
            case _:
                raise ValueError(f"Invalid split type: {split_type}")
        
        self.data = self._maybe_to_sparse(self.data)

    def _sample_negatives_per_group(
        self,
        exec_codes_positive: torch.Tensor,
        exec_codes_negative: torch.Tensor,
        node_edges_negative: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep at most n_pos_g * neg_sampling_ratio negatives per executionId."""
        if exec_codes_negative.numel() == 0:
            return exec_codes_negative, node_edges_negative

        unique_exec_pos, pos_counts = exec_codes_positive.unique(return_counts=True)

        neg_sort_idx = torch.argsort(exec_codes_negative, stable=True)
        sorted_exec_neg = exec_codes_negative[neg_sort_idx]

        unique_exec_neg, counts_neg = sorted_exec_neg.unique_consecutive(return_counts=True)

        pos_match = torch.searchsorted(unique_exec_pos, unique_exec_neg).clamp(0, len(unique_exec_pos) - 1)
        valid = unique_exec_pos[pos_match] == unique_exec_neg
        n_pos_per_group = torch.where(valid, pos_counts[pos_match], torch.zeros_like(counts_neg))
        keep_per_group = torch.minimum(n_pos_per_group * self.neg_sampling_ratio, counts_neg)

        offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts_neg.cumsum(0)[:-1]])
        within_pos = torch.arange(len(sorted_exec_neg)) - offsets.repeat_interleave(counts_neg)
        keep_mask = within_pos < keep_per_group.repeat_interleave(counts_neg)

        keep_idx = neg_sort_idx[keep_mask]
        return exec_codes_negative[keep_idx], node_edges_negative[:, keep_idx]

    def _canonical_ids(self, edges: torch.Tensor) -> torch.Tensor:
        src = torch.minimum(edges[0], edges[1])
        dst = torch.maximum(edges[0], edges[1])
        return src * self.num_nodes + dst

    def _remove_edges_from_graph(self, graph_edges: torch.Tensor, edges_to_remove: torch.Tensor) -> torch.Tensor:
        graph_ids = self._canonical_ids(graph_edges)
        remove_ids = self._canonical_ids(edges_to_remove).unique()
        keep_mask = ~torch.isin(graph_ids, remove_ids)
        return graph_edges[:, keep_mask] 

    def _maybe_to_sparse(self, data: Data) -> Data:
        """Convert edge_index to SparseTensor (CSR) format, unless running on MPS."""
        if self._use_sparse:
            data = self.to_sparse_tensor(data)
            data.edge_index = data.adj_t
        return data
    
    def __iter__(self) -> Iterator[Data]:
        return self.get_iterator()
    
    def _sample_train_edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        n_pos = self.pos_edges_train.shape[1]
        n_neg_sample = n_pos * self.neg_sampling_ratio
        neg_idx = torch.randperm(self.all_neg_edges_train.shape[1], device=self.device)[:n_neg_sample]
        sampled_neg = self.all_neg_edges_train[:, neg_idx]

        target_edges = torch.cat([self.pos_edges_train, sampled_neg], dim=1)
        labels = torch.cat([
            torch.ones(n_pos, device=self.device),
            torch.zeros(n_neg_sample, device=self.device),
        ], dim=0)

        perm = torch.randperm(target_edges.shape[1], device=self.device)
        return target_edges[:, perm], labels[perm]

    def __len__(self) -> int:
        if self.split_type == SplitType.TRAIN:
            # Number of negatives is capped by what is actually available (see _sample_train_edges).
            # __len__ MUST match the real batch count: Lightning triggers end-of-epoch validation at
            # batch index == num_training_batches, so an overestimate here means validation never fires.
            n_pos = self.pos_edges_train.shape[1]
            n = n_pos + min(n_pos * self.neg_sampling_ratio, self.all_neg_edges_train.shape[1])
        else:
            n = self.target_edges.shape[1]
        b = self.batch_size
        return max((n + b - 1) // b, 1)
        
    def get_iterator(self) -> Iterator[Data]:
        if self.split_type == SplitType.TRAIN:
            target_edges, labels = self._sample_train_edges()
            for start_idx in range(0, target_edges.shape[1], self.batch_size):
                end_idx = start_idx + self.batch_size
                data = Data(
                    x=self.data.x,
                    edge_index=self.data.edge_index,
                    target_edges=target_edges[:, start_idx:end_idx],
                    y=labels[start_idx:end_idx],
                )

                yield data

        else:
            for start_idx in range(0, self.target_edges.shape[1], self.batch_size):
                end_idx = start_idx + self.batch_size
                target_edges = self.target_edges[:, start_idx:end_idx]
                labels = self.labels[start_idx:end_idx]

                group_ids = self.group_ids[start_idx:end_idx] if self.group_ids is not None else None

                data = Data(
                    x=self.data.x,
                    edge_index=self.data.edge_index,
                    target_edges=target_edges,
                    y=labels,
                    group_ids=group_ids,
                )

                yield data

        
        







