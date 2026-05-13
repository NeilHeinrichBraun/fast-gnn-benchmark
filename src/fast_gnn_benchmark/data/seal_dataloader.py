from collections.abc import Iterator

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from fast_gnn_benchmark.data.dataset.seal import SEALDataset
from fast_gnn_benchmark.schemas.dataset_models import SplitType


class SEALDataLoader:
    def __init__(
        self,
        seal_dataset: SEALDataset,
        batch_size: int,
        split_type: SplitType,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        self.seal_dataset = seal_dataset
        self.batch_size = batch_size
        self.split_type = split_type

        effective_shuffle = shuffle and split_type == SplitType.TRAIN

        self._inner_loader = DataLoader(
            seal_dataset,
            batch_size=batch_size,
            shuffle=effective_shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def __iter__(self) -> Iterator[Data]:
        return iter(self._inner_loader)

    def __len__(self) -> int:
        return len(self._inner_loader)
