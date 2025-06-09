import os
import argparse
import requests
import numpy as np
import pandas as pd
import scanpy as sc
from io import BytesIO

def download_hubmap_h5ad(url: str, output_path: str, n_cells: int = 10000, seed: int = 42):
    print(f"[INFO] Downloading HuBMAP .h5ad dataset from:\n{url}")
    response = requests.get(url)
    response.raise_for_status()
    print("[INFO] Download complete. Reading data...")

    # Load directly from in-memory bytes
    adata = sc.read_h5ad(BytesIO(response.content))

    print(f"[INFO] Dataset has {adata.n_obs} cells and {adata.n_vars} genes")

    if n_cells < adata.n_obs:
        adata = adata[np.random.default_rng(seed).choice(adata.n_obs, n_cells, replace=False)]

    # Use raw if available
    if adata.raw is not None:
        adata = adata.raw.to_adata()

    X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    df = pd.DataFrame(X, columns=[f"Gene_{g}" for g in adata.var_names])

    # Add synthetic spatial coordinates
    coords = np.random.uniform(0, 1000, size=(adata.n_obs, 2))
    df["X_coord"] = coords[:, 0]
    df["Y_coord"] = coords[:, 1]

    # Add metadata if available
    for key in ["tissue", "donor_id", "cell_type"]:
        if key in adata.obs.columns:
            df[key] = adata.obs[key].values

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✓] Saved matrix to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download cell x gene matrix from HuBMAP .h5ad file")
    parser.add_argument("--url", type=str, required=True, help="Public HuBMAP .h5ad file URL")
    parser.add_argument("--output_path", type=str, default="data/hubmap_matrix.csv")
    parser.add_argument("--n_cells", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    download_hubmap_h5ad(args.url, args.output_path, args.n_cells, args.seed)
