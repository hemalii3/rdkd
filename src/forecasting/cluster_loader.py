"""
cluster_loader.py
-----------------
Load forecasting-ready cluster assignments saved by notebook 03.

Expected file types:
  - Feature+Ward (primary): feature_ward_k{k}_forecasting_handoff.csv
  - k-Shape (benchmark):    kshape_k{k}_forecasting_handoff.csv

These files are expected to be forecasting-ready, meaning:
  - one row per household
  - a 'cluster' column
  - all cluster labels are regular nonnegative integers

If negative labels are present, the handoff file is not forecasting-ready and
should be regenerated from the clustering notebook using the reassignment step.
"""

from pathlib import Path
import pandas as pd


def load_cluster_assignments(
    diagnostics_dir: Path,
    method: str,
    k: int,
) -> tuple[pd.Series, pd.Index]:
    """Load forecasting-ready cluster assignments for a given method and k.

    Parameters
    ----------
    diagnostics_dir : Path
        Path to results/diagnostics/.
    method : str
        "feature_ward" or "kshape".
    k : int
        Number of regular clusters.

    Returns
    -------
    cluster_series : pd.Series
        Series mapping household_id -> nonnegative cluster label (int).
    regular_ids : pd.Index
        Index of households with valid regular cluster assignments.

    Raises
    ------
    FileNotFoundError
        If the requested handoff file does not exist.
    ValueError
        If the file contains negative labels or missing labels.
    """
    diagnostics_dir = Path(diagnostics_dir)
    path = diagnostics_dir / f"{method}_k{k}_forecasting_handoff.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Cluster file not found: {path}\n"
            f"Run notebook 03 first with method='{method}' and k={k}."
        )

    df = pd.read_csv(path, index_col=0)

    col = "cluster" if "cluster" in df.columns else df.columns[0]
    if col != "cluster":
        print(f"  Warning: 'cluster' column not found in {path.name}; using '{col}'")

    cluster_series = pd.to_numeric(df[col], errors="coerce")

    if cluster_series.isna().any():
        bad_ids = cluster_series.index[cluster_series.isna()].tolist()[:10]
        raise ValueError(
            f"Forecasting handoff contains missing/non-numeric cluster labels. "
            f"Example household IDs: {bad_ids}"
        )

    if (cluster_series < 0).any():
        bad_ids = cluster_series.index[cluster_series < 0].tolist()[:10]
        raise ValueError(
            "Forecasting handoff contains negative cluster labels, but forecasting "
            "expects reassigned nonnegative labels only. "
            f"Example household IDs: {bad_ids}"
        )

    cluster_series = cluster_series.astype(int)
    cluster_series.name = "cluster"

    regular_ids = cluster_series.index

    print(f"  [{method} k={k}] {len(regular_ids):,} forecasting-ready households loaded")

    return cluster_series, regular_ids