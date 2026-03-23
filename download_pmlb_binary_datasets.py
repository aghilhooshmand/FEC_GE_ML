from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pmlb import classification_dataset_names, fetch_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _make_numeric_features(df_x: pd.DataFrame) -> pd.DataFrame:
    """Convert any categorical columns to numeric with one-hot encoding."""
    if df_x.empty:
        return df_x
    obj_cols = [c for c in df_x.columns if df_x[c].dtype == "object"]
    if obj_cols:
        df_x = pd.get_dummies(df_x, columns=obj_cols, dtype=float)
    for c in df_x.columns:
        df_x[c] = pd.to_numeric(df_x[c], errors="coerce")
    df_x = df_x.dropna(axis=0)
    return df_x


def _standardize_and_normalize(df_x: pd.DataFrame) -> pd.DataFrame:
    """
    Apply:
      1) z-score standardization
      2) min-max normalization to [0, 1]
    """
    if df_x.empty:
        return df_x
    x = df_x.to_numpy(dtype=float)
    x = StandardScaler().fit_transform(x)
    x = MinMaxScaler().fit_transform(x)
    return pd.DataFrame(x, columns=df_x.columns)


def _is_binary_target(y: pd.Series) -> bool:
    yy = y.dropna()
    if yy.empty:
        return False
    return yy.nunique() == 2


def download_and_prepare(
    out_dir: Path,
    local_cache_dir: Path | None = None,
    max_datasets: int | None = None,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n in classification_dataset_names if not n.startswith("_deprecated_")]
    names = sorted(set(names))
    if max_datasets is not None and max_datasets > 0:
        names = names[:max_datasets]

    saved: List[str] = []
    skipped: List[str] = []

    for name in names:
        try:
            df = fetch_data(name, local_cache_dir=str(local_cache_dir) if local_cache_dir else None)
        except Exception as exc:
            skipped.append(f"{name} (download error: {exc})")
            continue

        if "target" not in df.columns:
            skipped.append(f"{name} (missing target column)")
            continue

        y = df["target"]
        if not _is_binary_target(y):
            skipped.append(f"{name} (not binary)")
            continue

        x_raw = df.drop(columns=["target"])
        x_num = _make_numeric_features(x_raw)
        if x_num.empty:
            skipped.append(f"{name} (empty numeric features after cleanup)")
            continue

        # Align y to rows kept in x_num after numeric conversion/dropna.
        y = y.loc[x_num.index]
        if not _is_binary_target(y):
            skipped.append(f"{name} (target became non-binary after cleanup)")
            continue

        # Encode binary target to 0/1.
        classes = sorted(y.dropna().unique().tolist())
        class_to_int = {classes[0]: 0, classes[1]: 1}
        y_num = y.map(class_to_int).astype(int)

        x_scaled = _standardize_and_normalize(x_num)
        out_df = x_scaled.copy()
        out_df["target"] = y_num.to_numpy()

        out_path = out_dir / f"{name}.csv"
        out_df.to_csv(out_path, index=False)
        saved.append(name)
        print(f"Saved: {out_path} (rows={out_df.shape[0]}, cols={out_df.shape[1]})")

    # Save list of prepared datasets.
    manifest = {
        "count": len(saved),
        "datasets": saved,
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
    (out_dir / "pmlb_binary_datasets.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (out_dir / "pmlb_binary_datasets.txt").write_text(
        "\n".join(saved) + ("\n" if saved else ""),
        encoding="utf-8",
    )
    print(f"\nPrepared {len(saved)} binary datasets.")
    print(f"Saved list: {out_dir / 'pmlb_binary_datasets.txt'}")
    print(f"Saved manifest: {out_dir / 'pmlb_binary_datasets.json'}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download PMLB classification datasets, keep binary targets only, "
            "convert features to numeric, standardize + normalize, and save to CSV."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/pmlb",
        help="Output directory for prepared datasets.",
    )
    parser.add_argument(
        "--local-cache-dir",
        type=str,
        default="data/pmlb_cache",
        help="Local cache directory used by pmlb.fetch_data().",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Optional cap on number of dataset names to process.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.local_cache_dir) if args.local_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    download_and_prepare(
        out_dir=out_dir,
        local_cache_dir=cache_dir,
        max_datasets=args.max_datasets,
    )


if __name__ == "__main__":
    main()

