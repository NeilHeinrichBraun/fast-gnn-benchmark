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
    print("Databricks configured successfully")
except Exception as e:
    print(f"Could not configure Databricks: {e}")

import wandb
wandb.login()
print("wandb logged in successfully")


from pprint import pprint

from fast_gnn_benchmark.trainer import do_run, get_trainer_parameters_from_config

if __name__ == "__main__":
    duplicate_data_loader_parameters = True
    with wandb.init() as run:
        config = run.config

        base_config_file = config["base_config_file"]

        if duplicate_data_loader_parameters and "data_loader_parameters" in config:
            config["train_data_loader_parameters"] = config["data_loader_parameters"]
            config["val_data_loader_parameters"] = config["data_loader_parameters"]
            config["test_data_loader_parameters"] = config["data_loader_parameters"]

        base_config = get_trainer_parameters_from_config(base_config_file, config, import_global_config=False)  # type: ignore

        pprint(base_config.model_dump())

        do_run(base_config)
