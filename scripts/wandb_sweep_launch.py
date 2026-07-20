import subprocess
import os
import sys

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

import argparse
from pprint import pprint
from typing import Any

import yaml

from fast_gnn_benchmark.trainer import get_global_config


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        "-c",
        type=str,
        required=True,
        help="Path to the sweep config file. For example: configs/sweep/example.yml",
    )
    parser.add_argument("--project", "-p", type=str, required=True)
    parser.add_argument("--entity", "-e", type=str, required=True)
    return parser.parse_args()


def get_sweep_config(sweep_config_file: str) -> dict[str, Any]:
    with open(sweep_config_file, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    args = arg_parser()
    sweep_config_file = args.config_file
    project = args.project
    entity = args.entity

    sweep_config = get_sweep_config(sweep_config_file)
    pprint(sweep_config)
    sweep_id = wandb.sweep(
        sweep_config,
        project=project,
        entity=entity,
    )

    print()

    print("Now, we can launch the sweep:")
    subprocess.run([sys.executable, "-m", "wandb", "agent", f"{entity}/{project}/{sweep_id}"])
    
    print()

    print("To stop the sweep, run the following command:")
    print(f"{sys.executable} -m wandb sweep --stop {entity}/{project}/{sweep_id}")
