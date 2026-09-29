"""One-off script: build the small display-ready tables served to the deployed dashboard.

Reads the full datasets from the personal bucket, replicates the exact selection logic
of ds_dashboard/data_construction.py, and overwrites the shared dashboard prefix with
the reduced tables -- same table names, same columns, same dtypes.

No dashboard code is modified: data_construction.py keeps doing its own
drop_duplicates + sample(n=20) + metadata merge. On these reduced tables each subset
already holds exactly the 20 selected sessions, so that step becomes a no-op (it only
shuffles the display order), which also makes the result independent of SEED.

Usage:
    AWS_PROFILE=ds python build_dashboard_sample.py            # dry run, reads only
    AWS_PROFILE=ds python build_dashboard_sample.py --write    # overwrite the shared prefix

Re-run this script whenever SEED or SAMPLE_NUMBER changes in data_construction.py.
"""

import argparse

import awswrangler as wr
import numpy as np
import pandas as pd

SOURCE_BASE = "s3://mirakl-data-science-tmp2/nbraun/datasets/coview-mdm/"
DEST_BASE = "s3://mirakl-data-science-staging/dashboards_no_retention_rule/gnn_model_test/coview-mdm/"

# Must mirror ds_dashboard/data_construction.py
SEED = 56
SAMPLE_NUMBER = 20

NESTED_COLS = ["positives", "negatives", "products_returned"]


def read_table(base: str, name: str) -> pd.DataFrame:
    return wr.s3.read_parquet(f"{base}{name}.parquet/", path_suffix=".parquet")


def write_table(df: pd.DataFrame, name: str) -> None:
    wr.s3.to_parquet(
        df=df,
        path=f"{DEST_BASE}{name}.parquet/",
        dataset=True,
        mode="overwrite",
        index=False,
    )


def select_sessions(sessions_raw: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Same pipeline as data_construction.py: isin filter, then drop_duplicates, then sample."""
    subset = sessions_raw[sessions_raw["exec_code"].isin(results["exec_code"])]
    subset = subset.drop_duplicates(subset="trigger_internal_id", keep="first")
    return subset.sample(n=SAMPLE_NUMBER, random_state=SEED)


def iter_products(value):
    if isinstance(value, (list, np.ndarray)):
        yield from value


def collect_internal_ids(sessions: pd.DataFrame, results: list[pd.DataFrame]) -> set:
    """Every internal_id the dashboard will look up in product_metadata."""
    ids = set(sessions["trigger_internal_id"].tolist())

    for products in sessions["session_products"]:
        for product in iter_products(products):
            ids.add(product["internal_id"])

    for df in results:
        if "trigger_internal_id" in df.columns:
            ids.update(df["trigger_internal_id"].tolist())
        for col in NESTED_COLS:
            for products in df[col]:
                for product in iter_products(products):
                    ids.add(product["internal_id"])

    return ids


def simulate_page(
    sessions_raw, prod_results, model_results, metadata, label, source_ids
):
    """Replay data_construction.py for one page to prove nothing will crash.

    Mirrors the dashboard exactly: the displayed sessions are selected from the prod
    table only, then both result tables are attached with a left merge. A session
    absent from a result table is not an error -- the dashboard renders
    "Aucun produit renvoye" for it.
    """
    sessions = sessions_raw[sessions_raw["exec_code"].isin(prod_results["exec_code"])]
    sessions = sessions.drop_duplicates(subset="trigger_internal_id", keep="first")

    if len(sessions) < SAMPLE_NUMBER:
        raise AssertionError(
            f"[{label}] only {len(sessions)} distinct sessions available, "
            f"sample(n={SAMPLE_NUMBER}) would raise ValueError in the dashboard"
        )

    df_20 = sessions.sample(n=SAMPLE_NUMBER, random_state=SEED)

    lookup = metadata.set_index("internal_id")[["name", "imageUrl"]].to_dict(
        orient="index"
    )

    merged = df_20.merge(
        metadata,
        left_on="trigger_internal_id",
        right_on="internal_id",
        how="left",
    ).drop(columns="internal_id")
    if len(merged) != SAMPLE_NUMBER:
        raise AssertionError(
            f"[{label}] metadata merge changed the row count "
            f"({SAMPLE_NUMBER} -> {len(merged)}), duplicated internal_id in metadata"
        )

    missing = set()
    for products in df_20["session_products"]:
        for product in iter_products(products):
            if product["internal_id"] not in lookup:
                missing.add(product["internal_id"])

    print(f"  [{label}] {len(df_20)} sessions selected")

    for kind, results in [("prod", prod_results), ("model", model_results)]:
        reco = df_20[["exec_code"]].merge(results, on="exec_code", how="left")
        for col in NESTED_COLS:
            for products in reco[col]:
                for product in iter_products(products):
                    if product["internal_id"] not in lookup:
                        missing.add(product["internal_id"])

        returned = reco["products_returned"].apply(
            lambda p: len(list(iter_products(p)))
        )
        empty = int((returned == 0).sum())
        note = f", {empty} would show 'Aucun produit renvoye'" if empty else ""
        print(f"    {kind}: {returned.tolist()}{note}")

    # Missing from the reduced metadata but present at the source means the filtering
    # dropped something it should have kept. Missing at the source too is pre-existing
    # data, already rendered as "Produit inconnu" by app.py today.
    dropped = {internal_id for internal_id in missing if internal_id in source_ids}
    if dropped:
        raise AssertionError(
            f"[{label}] {len(dropped)} internal_id present at the source but missing "
            f"from the reduced product_metadata: {sorted(dropped)[:5]}"
        )
    if missing:
        print(
            f"    {len(missing)} internal_id have no metadata at the source either, "
            f"rendered as 'Produit inconnu' (unchanged from today)"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Overwrite the shared dashboard prefix. Off by default (dry run).",
    )
    args = parser.parse_args()

    print(f"Reading full datasets from {SOURCE_BASE}")
    sessions_raw = read_table(SOURCE_BASE, "sessions_raw_val")
    prod_results = read_table(SOURCE_BASE, "prod_results_val")
    model_results = read_table(SOURCE_BASE, "model_results_val")
    prod_neg_results = read_table(SOURCE_BASE, "prod_negative_only_results_val")
    model_neg_results = read_table(SOURCE_BASE, "model_negative_only_results_val")
    product_metadata = read_table(SOURCE_BASE, "product_metadata_val")

    source_dtypes = {
        "sessions_raw_val": sessions_raw.dtypes,
        "prod_results_val": prod_results.dtypes,
        "model_results_val": model_results.dtypes,
        "prod_negative_only_results_val": prod_neg_results.dtypes,
        "model_negative_only_results_val": model_neg_results.dtypes,
        "product_metadata_val": product_metadata.dtypes,
    }
    print(f"  sessions_raw_val: {len(sessions_raw)} rows")
    print(f"  product_metadata_val: {len(product_metadata)} rows")

    print("\nSelecting the sessions the dashboard will display")
    sessions_pos = select_sessions(sessions_raw, prod_results)
    sessions_neg = select_sessions(sessions_raw, prod_neg_results)

    # Single table feeding both pages: the dashboard re-derives each subset via isin.
    sessions_small = pd.concat([sessions_pos, sessions_neg])
    sessions_small = sessions_small[~sessions_small.index.duplicated(keep="first")]

    prod_results_small = prod_results[
        prod_results["exec_code"].isin(sessions_pos["exec_code"])
    ]
    model_results_small = model_results[
        model_results["exec_code"].isin(sessions_pos["exec_code"])
    ]
    prod_neg_small = prod_neg_results[
        prod_neg_results["exec_code"].isin(sessions_neg["exec_code"])
    ]
    model_neg_small = model_neg_results[
        model_neg_results["exec_code"].isin(sessions_neg["exec_code"])
    ]

    referenced_ids = collect_internal_ids(
        sessions_small,
        [prod_results_small, model_results_small, prod_neg_small, model_neg_small],
    )
    metadata_small = product_metadata[
        product_metadata["internal_id"].isin(referenced_ids)
    ]

    reduced = {
        "sessions_raw_val": sessions_small,
        "prod_results_val": prod_results_small,
        "model_results_val": model_results_small,
        "prod_negative_only_results_val": prod_neg_small,
        "model_negative_only_results_val": model_neg_small,
        "product_metadata_val": metadata_small,
    }

    print("\nReduced tables:")
    for name, df in reduced.items():
        print(f"  {name}: {len(df)} rows")

    print("\nSchema check against the source tables:")
    for name, df in reduced.items():
        diff = [
            f"{col}: {source_dtypes[name][col]} -> {df.dtypes[col]}"
            for col in df.columns
            if str(source_dtypes[name][col]) != str(df.dtypes[col])
        ]
        added = set(df.columns) - set(source_dtypes[name].index)
        removed = set(source_dtypes[name].index) - set(df.columns)
        if diff or added or removed:
            raise AssertionError(
                f"{name} schema drift: {diff or ''}{added or ''}{removed or ''}"
            )
        print(f"  {name}: identical ({len(df.columns)} columns)")

    source_ids = set(product_metadata["internal_id"].tolist())

    print("\nSimulating the dashboard on the reduced tables:")
    simulate_page(
        sessions_small,
        prod_results_small,
        model_results_small,
        metadata_small,
        "app.py",
        source_ids,
    )
    simulate_page(
        sessions_small,
        prod_neg_small,
        model_neg_small,
        metadata_small,
        "2_negative_only.py",
        source_ids,
    )

    if not args.write:
        print(
            f"\nDry run: nothing written. Re-run with --write to overwrite {DEST_BASE}"
        )
        return

    print(f"\nOverwriting {DEST_BASE}")
    for name, df in reduced.items():
        write_table(df, name)
        print(f"  wrote {name} ({len(df)} rows)")

    print("\nReading back from the destination and re-checking:")
    for name, df in reduced.items():
        back = read_table(DEST_BASE, name)
        if len(back) != len(df):
            raise AssertionError(f"{name}: wrote {len(df)} rows, read back {len(back)}")
        diff = [
            f"{col}: {source_dtypes[name][col]} -> {back.dtypes[col]}"
            for col in back.columns
            if str(source_dtypes[name][col]) != str(back.dtypes[col])
        ]
        if diff or set(back.columns) != set(source_dtypes[name].index):
            raise AssertionError(f"{name} schema drift after round-trip: {diff}")
        print(f"  {name}: {len(back)} rows, schema identical")

    simulate_page(
        read_table(DEST_BASE, "sessions_raw_val"),
        read_table(DEST_BASE, "prod_results_val"),
        read_table(DEST_BASE, "model_results_val"),
        read_table(DEST_BASE, "product_metadata_val"),
        "round-trip app.py",
        source_ids,
    )

    print("\nDone. Restart the dashboard pod to pick up the new data.")


if __name__ == "__main__":
    main()
