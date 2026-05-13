import pathlib

import torch
from torch_geometric.data import Data

DATASET_FILENAME = "amazon_products.pt"


class AmazonDataset:
    def __init__(self, root: str = "./datasets/amazon"):
        self.root = pathlib.Path(root)

        payload = torch.load(
            self.root / DATASET_FILENAME,
            map_location="cpu",
            weights_only=False,
        )

        self.data = payload["data"]
        self.num_nodes = payload["num_nodes"]
        self.split = payload["split"]

    def __len__(self):
        return 1

    def __getitem__(self, index: int) -> Data:
        assert index == 0, "Index must be 0"
        return self.data
