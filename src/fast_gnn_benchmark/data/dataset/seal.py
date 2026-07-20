from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import k_hop_subgraph, negative_sampling, to_scipy_sparse_matrix

from fast_gnn_benchmark.data.dataset.seal_cpp import (
    AVAILABLE as _CPP_AVAILABLE,
    batch_extract as _cpp_batch_extract,
    build_csr as _cpp_build_csr,
)

def drnl_node_labeling(edge_index, src, dst, num_nodes=None) -> torch.Tensor:
    src, dst = (dst, src) if src > dst else (src, dst)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def zero_node_labeling(num_nodes: int) -> torch.Tensor:
    return torch.zeros(num_nodes, dtype=torch.long)


class SEALDataset(InMemoryDataset):
    def __init__(
        self,
        dataset,
        split: str,
        num_hops: int,
        node_labeling: str,
        use_features: bool,
        max_nodes_per_hop: int | None,
        root: str | None = None,
        num_negative_train: int | None = None,
        num_train_samples: int | None = None,
        use_cpp_extension: bool = True,
    ):
        self._dataset = dataset
        self._data = dataset[0]
        self.num_hops = num_hops
        self.node_labeling = node_labeling
        self.use_features = use_features
        self.max_nodes_per_hop = max_nodes_per_hop
        self.num_negative_train = num_negative_train
        self.num_train_samples = num_train_samples
        self.dataset_name = getattr(dataset, "name", type(dataset).__name__)
        self.use_cpp = _CPP_AVAILABLE and use_cpp_extension

        self._split = split
        self._split_idx = ["train", "valid", "test"].index(split)

        resolved_root = root if root is not None else dataset.root

        super().__init__(resolved_root)
        self.load(self.processed_paths[self._split_idx])

    @property
    def processed_file_names(self) -> list[str]:
        n = f"n{self.num_train_samples}" if self.num_train_samples is not None else "nall"
        suffix = f"{self.dataset_name}_h{self.num_hops}_{self.node_labeling}_feat{int(self.use_features)}_{n}"
        return [
            f"SEAL_{suffix}_train.pt",
            f"SEAL_{suffix}_valid.pt",
            f"SEAL_{suffix}_test.pt",
        ]

    def process(self):
        full_edge_index = self._data.edge_index
        num_nodes = self._data.num_nodes

        if self.use_cpp:
            self._row_ptr, self._col_idx = _cpp_build_csr(full_edge_index, num_nodes)

        assert hasattr(self._dataset, "split"), (
            "Only OGBL datasets (with .split) are supported for now"
        )
        splits = self._dataset.split

        train_pos = splits["train"]["edge"].T
        if self.num_train_samples is not None and self.num_train_samples < train_pos.shape[1]:
            idx = torch.randperm(train_pos.shape[1])[:self.num_train_samples]
            train_pos = train_pos[:, idx]
        n_neg = (
            self.num_negative_train
            if self.num_negative_train is not None
            else train_pos.shape[1]
        )
        train_neg = negative_sampling(
            full_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=n_neg,
            force_undirected=True,
        )

        val_pos  = splits["valid"]["edge"].T
        val_neg  = splits["valid"]["edge_neg"].T
        test_pos = splits["test"]["edge"].T
        test_neg = splits["test"]["edge_neg"].T

        train_pos_list = self.extract_enclosing_subgraphs(full_edge_index, train_pos, y=1, desc="train_pos")
        train_neg_list = self.extract_enclosing_subgraphs(full_edge_index, train_neg, y=0, desc="train_neg")
        val_pos_list   = self.extract_enclosing_subgraphs(full_edge_index, val_pos,   y=1, desc="val_pos")
        val_neg_list   = self.extract_enclosing_subgraphs(full_edge_index, val_neg,   y=0, desc="val_neg")
        test_pos_list  = self.extract_enclosing_subgraphs(full_edge_index, test_pos,  y=1, desc="test_pos")
        test_neg_list  = self.extract_enclosing_subgraphs(full_edge_index, test_neg,  y=0, desc="test_neg")

        all_lists = [
            train_pos_list, train_neg_list,
            val_pos_list, val_neg_list,
            test_pos_list, test_neg_list,
        ]
        max_z = max(int(d.z.max()) for lst in all_lists for d in lst)

        for data in chain(*all_lists):
            z_one_hot = F.one_hot(data.z, max_z + 1).float()
            if self.use_features and data.x is not None:
                data.x = torch.cat([z_one_hot, data.x], dim=-1)
            else:
                data.x = z_one_hot

        self.save(train_pos_list + train_neg_list, self.processed_paths[0])
        self.save(val_pos_list   + val_neg_list,   self.processed_paths[1])
        self.save(test_pos_list  + test_neg_list,  self.processed_paths[2])

    def extract_enclosing_subgraphs(
        self, edge_index, edge_label_index, y, desc: str = ""
    ) -> list[Data]:
        if self.use_cpp:
            return self._extract_cpp(edge_label_index, y, desc)
        return self._extract_python(edge_index, edge_label_index, y, desc)


    def _extract_cpp(self, edge_label_index, y, desc: str) -> list[Data]:
        N = edge_label_index.shape[1]
        print(f"[SEAL C++] {desc}: {N} pairs", flush=True)

        sub_edges, z_labels, sub_nodes_list = _cpp_batch_extract(
            self._row_ptr,
            self._col_idx,
            edge_label_index[0].contiguous(),
            edge_label_index[1].contiguous(),
            self.num_hops,
            self._data.num_nodes,
        )

        source_x = self._data.x
        return [
            Data(
                x=source_x[sub_nodes_list[i]] if self.use_features else None,
                z=z_labels[i],
                edge_index=sub_edges[i],
                y=y,
            )
            for i in range(N)
        ]

    def _extract_python(self, edge_index, edge_label_index, y, desc: str) -> list[Data]:
        source_x = self._data.x
        data_list = []
        pairs = edge_label_index.t().tolist()

        for src, dst in tqdm(pairs, desc=f"[SEAL] {desc}" if desc else "[SEAL] extracting"):
            sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
                [src, dst], self.num_hops, edge_index, relabel_nodes=True
            )
            src, dst = mapping.tolist()

            mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
            mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
            sub_edge_index = sub_edge_index[:, mask1 & mask2]

            match self.node_labeling:
                case "drnl":
                    z = drnl_node_labeling(sub_edge_index, src, dst, num_nodes=sub_nodes.size(0))
                case "zero":
                    z = zero_node_labeling(sub_nodes.size(0))
                case _:
                    raise ValueError(f"Unknown node_labeling: {self.node_labeling}")

            x_sub = source_x[sub_nodes] if self.use_features else None
            data_list.append(Data(x=x_sub, z=z, edge_index=sub_edge_index, y=y))

        return data_list
