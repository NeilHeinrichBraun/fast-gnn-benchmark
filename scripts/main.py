import subprocess
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

BUNDLE_ROOT = "/Workspace/Users/neil.braun@mirakl.com/.bundle/fast-gnn-benchmark/dev/files"

subprocess.run(["pip", "install", BUNDLE_ROOT], check=True)

os.chdir("/tmp")

try:
    from pyspark.dbutils import DBUtils
    from pyspark import SparkContext
    dbutils = DBUtils(SparkContext.getOrCreate())
    wandb_key = dbutils.secrets.get(scope="nbraun", key="wandb_api_key")
    os.environ["WANDB_API_KEY"] = wandb_key
    os.environ["WANDB_DIR"] = "/tmp"
    os.environ["WANDB_CACHE_DIR"] = "/tmp"
    os.environ["WANDB_DATA_DIR"] = "/tmp"
    os.environ["WANDB_CONFIG_DIR"] = "/tmp"

    import wandb
    wandb.login(key=wandb_key)
    print("wandb logged in successfully")
except Exception as e:
    print(f"Could not configure wandb: {e}")

from pprint import pprint
from typing import Any

from fast_gnn_benchmark.trainer import do_run, fix_seed, get_trainer_parameters_from_config


def main(file_path: str, override_dict: dict[str, Any] = {}) -> None:
    trainer_parameters = get_trainer_parameters_from_config(file_path, override_dict, import_global_config=False)

    pprint(trainer_parameters.model_dump())
    print()

    if trainer_parameters.seed is not None:
        fix_seed(trainer_parameters.seed)
    else:
        print("No seed provided")

    test_metrics = do_run(trainer_parameters)

    for data_loader_idx in range(len(test_metrics)):
        print(f"Results for data loader {data_loader_idx}:")
        pprint(test_metrics[data_loader_idx])
        print()


if __name__ == "__main__":
    import argparse

    from fast_gnn_benchmark.utils import recursive_defaultdict

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", type=str, required=False, default="configs/ogbn_products/sage.yml")
    parser.add_argument("--epochs", type=int, required=False, default=None)
    parser.add_argument("--lr", type=float, required=False, default=None)
    parser.add_argument("--train_parts", type=int, required=False, default=None)
    parser.add_argument("--val_test_parts", type=int, required=False, default=None)
    parser.add_argument("--drop_edge_ratio", type=float, required=False, default=None)
    parser.add_argument("--seed", type=int, required=False, default=None)
    parser.add_argument("--tag", type=str, required=False, default=None)
    parser.add_argument("--batch_size", type=int, required=False, default=None)
    args = parser.parse_args()

    override_dict = recursive_defaultdict()
    if args.train_parts is not None:
        override_dict["data_parameters"]["train_data_loader_parameters"]["num_parts"] = args.train_parts

    if args.drop_edge_ratio is not None:
        override_dict["data_parameters"]["train_data_loader_parameters"]["drop_edge_ratio"] = args.drop_edge_ratio

    if args.val_test_parts is not None:
        override_dict["data_parameters"]["val_data_loader_parameters"]["num_parts"] = args.val_test_parts
        override_dict["data_parameters"]["test_data_loader_parameters"]["num_parts"] = args.val_test_parts

    if args.lr is not None:
        override_dict["model_parameters"]["optimizer"]["parameters"]["lr"] = args.lr

    if args.seed is not None:
        override_dict["seed"] = args.seed

    if args.epochs is not None:
        override_dict["trainer_config"]["max_epochs"] = args.epochs

    if args.tag is not None:
        override_dict["wandb_logger_parameters"]["tags"] = [args.tag]

    if args.batch_size is not None:
        override_dict["data_parameters"]["train_data_loader_parameters"]["batch_size"] = args.batch_size

    file_path = args.config_file

    main(file_path, override_dict)
