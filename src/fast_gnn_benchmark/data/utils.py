from typing import Any

import torch
from torch_geometric import utils
from torch_geometric.data import Data

def print_data_properties_link_prediction(dataset: Any) -> None:
    data = dataset[0]
    print()
    print("Link prediction dataset properties:")
    print()

    print("Number of nodes:", data.x.shape[0])  # type: ignore
    print("Number of train edges (with repetition):", data.edge_index.shape[1])  # type: ignore
    print("Number of features:", data.x.shape[1])  # type: ignore
    print()

    print("Train edges:", dataset.split["train"]["edge"].shape[0])
    print("Valid edges:", dataset.split["valid"]["edge"].shape[0])
    print("Negative valid edges:", dataset.split["valid"]["edge_neg"].shape[0])
    print("Test edges:", dataset.split["test"]["edge"].shape[0])
    print("Negative test edges:", dataset.split["test"]["edge_neg"].shape[0])


def remove_duplicate_edges(edges: torch.Tensor) -> torch.Tensor:
    """
    Remove duplicate edges from the edge index tensor.

    Args:
        edges (torch.Tensor): The edge index tensor of shape (2, num_edges).

    Returns:
        torch.Tensor: A Tensor containing the unique edges.
    """
    return torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), dtype=torch.int32).coalesce().indices()


def remove_self_loops(edges: torch.Tensor) -> torch.Tensor:
    """
    Remove self-loops from the edge index tensor.

    Args:
        edges (torch.Tensor): The edge index tensor of shape (2, num_edges).

    Returns:
        torch.Tensor: The edge index tensor with self-loops removed.
    """

    sparse_coo = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), dtype=torch.int32).coalesce()
    self_loops_mask = sparse_coo.indices()[0] == sparse_coo.indices()[1]
    sparse_coo_without_self_loops = torch.sparse_coo_tensor(
        sparse_coo.indices()[:, ~self_loops_mask], sparse_coo.values()[~self_loops_mask], dtype=torch.int32
    ).coalesce()

    return sparse_coo_without_self_loops.indices()


def add_self_loops_and_remove_duplicate_edges(edges: torch.Tensor) -> torch.Tensor:
    """
    Add self-loops to the edge index tensor. It also removes duplicate edges if any.

    Args:
        edges (torch.Tensor): The edge index tensor of shape (2, num_edges).

    Returns:
        torch.Tensor: The edge index tensor with self-loops added.
    """
    sparse_coo = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), dtype=torch.int32).coalesce()
    size = sparse_coo.size()

    indices = torch.arange(size[0]).expand(2, -1)
    sparse_coo_with_self_loops = torch.sparse_coo_tensor(
        torch.cat([indices, sparse_coo.indices()], dim=1),
        torch.cat([torch.ones(size[0]), sparse_coo.values()], dim=0),
        dtype=torch.int32,
    ).coalesce()

    return sparse_coo_with_self_loops.indices()


def to_undirected(edges: torch.Tensor) -> torch.Tensor:
    """
    Convert the edge index tensor to an undirected graph.
    """
    return torch.cat([edges, edges.flip(0)], dim=1)
