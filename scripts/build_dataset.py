import subprocess
import sys

BUNDLE_ROOT = "/Workspace/Users/neil.braun@mirakl.com/.bundle/fast-gnn-benchmark/dev/files"


def main(config_file: str) -> None:
    """Installe le package du bundle, puis construit le dataset.

    Les imports sont locaux : le package n'est disponible qu'après le pip install.

    stdout est repassé en line buffering : redirigé vers un fichier par le job Databricks, il
    serait sinon bufferisé par blocs et les logs n'apparaîtraient que par paquets.

    Args:
        config_file: Chemin du YAML de configuration du pipeline.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    subprocess.run(["pip", "install", BUNDLE_ROOT], check=True)

    from pprint import pprint

    from pyspark.sql import SparkSession

    from fast_gnn_benchmark.data.pipeline.runner import do_build, get_pipeline_parameters_from_config

    parameters = get_pipeline_parameters_from_config(config_file)

    pprint(parameters.model_dump())
    print()

    data = do_build(SparkSession.builder.getOrCreate(), parameters)

    print(data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()

    main(args.config_file)
