import torch
from torch_geometric.data import Data


def build_pyg_data(
    num_nodes: int,
    train_edge_index: torch.Tensor,
    train_neg_edge_index: torch.Tensor,
    train_pos_edge_index: torch.Tensor,
    train_edge_attr: torch.Tensor,
    val_pos_edge_index: torch.Tensor,
    val_neg_edge_index: torch.Tensor,
    test_pos_edge_index: torch.Tensor,
    test_neg_edge_index: torch.Tensor,
    x: torch.Tensor | None = None,
) -> Data:
    """
    Assemble all tensors into a single PyG Data object.
    """
    data = Data(
        edge_index=train_edge_index,
        edge_attr=train_edge_attr,
        num_nodes=num_nodes,
    )

    if x is not None:
        data.x = x

    data.train_neg_edge_index = train_neg_edge_index
    data.train_pos_edge_index = train_pos_edge_index

    data.val_pos_edge_index = val_pos_edge_index
    data.val_neg_edge_index = val_neg_edge_index
    data.test_pos_edge_index = test_pos_edge_index
    data.test_neg_edge_index = test_neg_edge_index

    return data
