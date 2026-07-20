import boto3
import io

import torch
from torch_geometric.data import Data


class CoViewMDMDataset:
    def __init__(self, bucket: str =  "mirakl-data-science-tmp2", s3_key: str = "nbraun/datasets/coview-mdm/data.pt"):
        
        s3 = boto3.client("s3")

        loaded = torch.load(io.BytesIO(s3.get_object(Bucket=bucket, Key=s3_key)["Body"].read()), weights_only=False)
    
        raw = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        data = Data(**raw) if isinstance(raw, dict) else raw

        self.data = Data(x=data.x, edge_index=data.edge_index)
        self.num_nodes = int(data.num_nodes)
        self.split = {
            "train": {"edge": data.train_pos_edge_index.T, "edge_neg": data.train_neg_edge_index.T},
            "valid": {"edge": data.val_pos_edge_index.T, "edge_neg": data.val_neg_edge_index.T},
            "test":  {"edge": data.test_pos_edge_index.T, "edge_neg": data.test_neg_edge_index.T},
        }

    def __len__(self):
        return 1

    def __getitem__(self, index: int) -> Data:
        assert index == 0, "Index must be 0"
        return self.data
