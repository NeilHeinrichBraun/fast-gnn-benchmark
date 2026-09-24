from pydantic import BaseModel, field_validator

from fast_gnn_benchmark.data.dataset.coview_mdm import CoViewMDMDataset
from fast_gnn_benchmark.data.link_dataloader import LinkLoader
from fast_gnn_benchmark.data.trigger_dataloader import TriggerLoader
from fast_gnn_benchmark.data.utils import (
    add_self_loops_and_remove_duplicate_edges,
    print_data_properties_link_prediction,
    remove_duplicate_edges,
    remove_self_loops,
    to_undirected,
)
from fast_gnn_benchmark.schemas.dataset_models import (
    DataLoaderParametersChoices,
    DataLoaderType,
    DatasetType,
    SplitType,
)

DatasetTypeChoices = CoViewMDMDataset

DataLoaderTypeChoices = LinkLoader | TriggerLoader


class DataParameters(BaseModel):
    dataset_type: DatasetType
    to_undirected: bool = False
    add_self_loops_and_remove_duplicate_edges: bool = False
    remove_duplicate_edges: bool = False
    remove_self_loops: bool = False

    train_data_loader_parameters: DataLoaderParametersChoices
    val_data_loader_parameters: DataLoaderParametersChoices
    test_data_loader_parameters: DataLoaderParametersChoices

    @field_validator(
        "train_data_loader_parameters",
        "val_data_loader_parameters",
        "test_data_loader_parameters",
        mode="before",
    )
    @classmethod
    def convert_data_loader_type(cls, v):
        if isinstance(v, dict) and "data_loader_type" in v:
            data_loader_type = v["data_loader_type"]
            if isinstance(data_loader_type, str):
                try:
                    v = v.copy()  # Don't modify the original
                    v["data_loader_type"] = DataLoaderType(data_loader_type)
                except ValueError:
                    raise ValueError(
                        f"Invalid data_loader_type: {data_loader_type}. Must be one of: {[e.value for e in DataLoaderType]}"
                    ) from None
        return v

    def get_dataset(self) -> DatasetTypeChoices:
        match self.dataset_type:
            case DatasetType.COVIEW_MDM:
                dataset = CoViewMDMDataset(
                    bucket="mirakl-data-science-tmp2", s3_key="nbraun/datasets/coview-mdm/data.pt"
                )

            case DatasetType.COVIEW_MDM_PROTOTYPE:
                dataset = CoViewMDMDataset(
                    bucket="mirakl-data-science-tmp2", s3_key="nbraun/datasets/coview-mdm/data_prototype.pt"
                )

            case _:
                raise ValueError(f"Invalid dataset type: {self}")

        assert not (
            self.add_self_loops_and_remove_duplicate_edges and self.remove_duplicate_edges
        ), "Cannot add self-loops and remove duplicate edges at the same time"

        if self.to_undirected:
            print(f"Converting to undirected graph for {self.dataset_type}")
            dataset[0].edge_index = to_undirected(dataset[0].edge_index)  # type: ignore

        if self.add_self_loops_and_remove_duplicate_edges:
            print(f"Adding self-loops and removing duplicate edges for {self.dataset_type}")
            dataset[0].edge_index = add_self_loops_and_remove_duplicate_edges(dataset[0].edge_index)  # type: ignore

        if self.remove_duplicate_edges:
            print(f"Removing duplicate edges for {self.dataset_type}")
            dataset[0].edge_index = remove_duplicate_edges(dataset[0].edge_index)  # type: ignore

        if self.remove_self_loops:
            print(f"Removing self-loops for {self.dataset_type}")
            dataset[0].edge_index = remove_self_loops(dataset[0].edge_index)  # type: ignore

        print_data_properties_link_prediction(dataset)

        return dataset

    def get_data_loader(
        self, dataset: DatasetTypeChoices, split_type: SplitType, data_loader_parameters: DataLoaderParametersChoices
    ) -> DataLoaderTypeChoices:
        match data_loader_parameters.data_loader_type:
            case DataLoaderType.LINK_LOADER:
                return LinkLoader(
                    dataset,
                    batch_size=data_loader_parameters.batch_size,
                    mask_loss_edges=data_loader_parameters.mask_loss_edges,
                    max_rejection_sampling_iterations=data_loader_parameters.max_rejection_sampling_iterations,
                    negative_sampling_ratio=data_loader_parameters.negative_sampling_ratio,
                    on_device=data_loader_parameters.on_device,
                    split_type=split_type,
                    use_val_edges_as_input=data_loader_parameters.use_val_edges_as_input,
                    use_precomputed_negatives=data_loader_parameters.use_precomputed_negatives,
                    precomputed_negatives_sampling_ratio=data_loader_parameters.precomputed_negatives_sampling_ratio,
                )

            case DataLoaderType.TRIGGER_LOADER:
                return TriggerLoader(
                    dataset,
                    batch_size=data_loader_parameters.batch_size,
                    mask_loss_edges=data_loader_parameters.mask_loss_edges,
                    on_device=data_loader_parameters.on_device,
                    split_type=split_type,
                    use_val_edges_as_input=data_loader_parameters.use_val_edges_as_input,
                    neg_sampling_ratio=data_loader_parameters.neg_sampling_ratio,
                )

            case _:
                raise ValueError(f"Invalid data loader type: {data_loader_parameters.data_loader_type}")

    def get(
        self,
    ) -> tuple[DataLoaderTypeChoices, DataLoaderTypeChoices, DataLoaderTypeChoices]:
        dataset = self.get_dataset()

        train_data_loader = self.get_data_loader(dataset, SplitType.TRAIN, self.train_data_loader_parameters)
        val_data_loader = self.get_data_loader(dataset, SplitType.VAL, self.val_data_loader_parameters)
        test_data_loader = self.get_data_loader(dataset, SplitType.TEST, self.test_data_loader_parameters)

        return train_data_loader, val_data_loader, test_data_loader
