import os

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

SOURCE_PATH = "/Users/neil.braun/Desktop/baseline_AR2023/amazon_graph.pt"
OUTPUT_DIR = "./datasets/amazon"
OUTPUT_FILE = "amazon_products.pt"
NUM_VAL = 0.1
NUM_TEST = 0.1
NEG_SAMPLING_RATIO = 1.0
SEED = 42


def load_graph(source_path: str) -> Data:
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    graph = checkpoint["graph"]

    assert graph.x.shape == (139998, 4096), f"Unexpected x shape: {graph.x.shape}"
    assert graph.edge_index.shape == (2, 252230), f"Unexpected edge_index shape: {graph.edge_index.shape}"
    assert graph.num_nodes == 139998, f"Unexpected num_nodes: {graph.num_nodes}"

    if graph.x.dtype != torch.float32:
        print(f"[WARN] x dtype is {graph.x.dtype}, casting to float32")
        graph.x = graph.x.to(torch.float32)

    return graph


def build_splits(
    graph: Data,
    num_val: float,
    num_test: float,
    seed: int,
    neg_sampling_ratio: float,
) -> tuple[Data, dict]:
    torch.manual_seed(seed)

    transform = RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=neg_sampling_ratio,
    )
    train_data, val_data, test_data = transform(graph)

    split = {
        "train": {
            "edge": train_data.edge_label_index.T,
        },
        "valid": {
            "edge": val_data.edge_label_index.T[val_data.edge_label == 1],
            "edge_neg": val_data.edge_label_index.T[val_data.edge_label == 0],
        },
        "test": {
            "edge": test_data.edge_label_index.T[test_data.edge_label == 1],
            "edge_neg": test_data.edge_label_index.T[test_data.edge_label == 0],
        },
    }

    data = Data(
        x=graph.x,
        edge_index=train_data.edge_index,
        num_nodes=graph.num_nodes,
    )

    return data, split


def save_dataset(data: Data, split: dict, output_dir: str, output_file: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_file)
    torch.save({"data": data, "split": split, "num_nodes": data.num_nodes}, output_path)

    print(f"Saved to           : {output_path}")
    print(f"Nodes              : {data.num_nodes}")
    print(f"Message-pass edges : {data.edge_index.shape[1]}")
    print(f"Feature dim        : {data.x.shape[1]}")
    print(f"x dtype            : {data.x.dtype}")
    print()
    print(f"Train edges        : {split['train']['edge'].shape}")
    print(f"Val   pos          : {split['valid']['edge'].shape}")
    print(f"Val   neg          : {split['valid']['edge_neg'].shape}")
    print(f"Test  pos          : {split['test']['edge'].shape}")
    print(f"Test  neg          : {split['test']['edge_neg'].shape}")


def main() -> None:
    graph = load_graph(SOURCE_PATH)
    data, split = build_splits(graph, NUM_VAL, NUM_TEST, SEED, NEG_SAMPLING_RATIO)
    save_dataset(data, split, OUTPUT_DIR, OUTPUT_FILE)


if __name__ == "__main__":
    main()
