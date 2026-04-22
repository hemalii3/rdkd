"""
cluster_loader.py
-----------------
Load cluster assignments saved by notebook 03.

The clustering notebook saves two sets of files:
  - Feature+Ward (primary) : feature_ward_k{k}_forecasting_handoff.csv
  - k-Shape (benchmark)    : kshape_k{k}_forecasting_handoff.csv

Both have the same format: one row per household, with a 'cluster' column.
Regular clusters are labelled 0..k-1. Degenerate households are labelled < 0.
"""

from pathlib import Path
import pandas as pd


def load_cluster_assignments(
    diagnostics_dir: Path,
    method: str,
    k: int,
) -> tuple:
    """Load cluster assignments for a given method and k.

    Parameters
    ----------
    diagnostics_dir : Path to results/diagnostics/.
    method          : "feature_ward" or "kshape".
    k               : Number of regular clusters.

    Returns
    -------
    cluster_series  : Series mapping household_id -> cluster label (int).
    degenerate_ids  : Index of households with labels < 0.
    regular_ids     : Index of households with labels >= 0.
    """
    diagnostics_dir = Path(diagnostics_dir)
    path = diagnostics_dir / f"{method}_k{k}_forecasting_handoff.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Cluster file not found: {path}\n"
            f"Run notebook 03 first with method='{method}' and k={k}."
        )

    df = pd.read_csv(path, index_col=0)

    # Find the cluster column
    col = "cluster" if "cluster" in df.columns else df.columns[0]
    if col != "cluster":
        print(f"  Warning: 'cluster' column not found in {path.name}; using '{col}'")

    cluster_series      = df[col].astype(int)
    cluster_series.name = "cluster"

    regular_ids    = cluster_series[cluster_series >= 0].index
    degenerate_ids = cluster_series[cluster_series <  0].index

    print(f"  [{method} k={k}] {len(regular_ids):,} regular, "
          f"{len(degenerate_ids):,} degenerate households")

    return cluster_series, degenerate_ids, regular_ids
