from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class SplitType(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetType(Enum):
    COVIEW_MDM = "coview-mdm"
    COVIEW_MDM_PROTOTYPE = "coview-mdm-prototype"


class DataLoaderType(Enum):
    LINK_LOADER = "link_loader"
    TRIGGER_LOADER = "trigger_loader"


class DataLoaderParameters(BaseModel):
    pass


class LinkLoaderParameters(DataLoaderParameters):
    data_loader_type: Literal[DataLoaderType.LINK_LOADER] = DataLoaderType.LINK_LOADER
    batch_size: int
    mask_loss_edges: bool = True
    max_rejection_sampling_iterations: int = 3
    negative_sampling_ratio: float = 0.5
    on_device: bool = True

    use_val_edges_as_input: bool = False
    use_precomputed_negatives: bool = False
    precomputed_negatives_sampling_ratio: float | None = None


class TriggerLoaderParameters(DataLoaderParameters):
    data_loader_type: Literal[DataLoaderType.TRIGGER_LOADER] = DataLoaderType.TRIGGER_LOADER
    batch_size: int
    mask_loss_edges: bool = True
    on_device: bool = True
    use_val_edges_as_input: bool = False
    neg_sampling_ratio: int = 5


DataLoaderParametersChoices = Annotated[
    LinkLoaderParameters | TriggerLoaderParameters,
    Field(discriminator="data_loader_type"),
]
