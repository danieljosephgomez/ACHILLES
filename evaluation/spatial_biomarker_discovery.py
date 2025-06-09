import argparse
import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser(description="Spatial biomarker discovery and phenotyping for spatial omics data.")
    parser.add_argument("--input_h5ad", type=str, required=True, help="Path to input AnnData (.h5ad) file.")
    parser.add_argument("--label_key", type=str, default="cell_type", help="AnnData.obs key for cell/tissue labels.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    adata = sc.read_h5ad(args.input_h5ad)

    # --- 1. Spatial Biomarker Discovery (Squidpy) ---
    print("[INFO] Running spatially variable gene detection (Squidpy)...")
    sq.gr.spatial_neighbors(adata)
    sq.gr.spatial_autocorr(adata, mode="moran")
    moran_df = adata.uns["moranI"]
    moran_df.to_csv(os.path.join(args.output_dir, "spatially_variable_genes.csv"))
    print(f"[✓] Moran's I results saved to {args.output_dir}/spatially_variable_genes.csv")

    # --- 2. Label-efficient Cell Phenotyping (Semi-supervised) ---
    print("[INFO] Running label-efficient cell phenotyping (Scanpy)...")
    # Example: Use partial labels for training a classifier
    from sklearn.semi_supervised import LabelSpreading
    X = adata.obsm["X_pca"] if "X_pca" in adata.obsm else sc.tl.pca(adata, copy=True).obsm["X_pca"]
    y = adata.obs[args.label_key].copy()
    # Mask most labels as -1 (unlabeled)
    mask = np.random.rand(len(y)) < 0.8
    y_masked = y.copy()
    y_masked[mask] = -1
    label_encoder = {k: i for i, k in enumerate(y.unique())}
    y_encoded = y.map(label_encoder)
    y_encoded[mask] = -1
    clf = LabelSpreading()
    clf.fit(X, y_encoded)
    adata.obs["label_spread"] = pd.Series(clf.transduction_).map({v: k for k, v in label_encoder.items()})
    adata.obs[["label_spread"]].to_csv(os.path.join(args.output_dir, "label_efficient_cell_phenotyping.csv"))
    print(f"[✓] Label-efficient cell phenotyping results saved to {args.output_dir}/label_efficient_cell_phenotyping.csv")

    # --- 3. Unsupervised Tissue Phenotyping (Clustering) ---
    print("[INFO] Running unsupervised tissue phenotyping (Scanpy)...")
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, key_added="unsupervised_tissue_clusters")
    adata.obs[["unsupervised_tissue_clusters"]].to_csv(os.path.join(args.output_dir, "unsupervised_tissue_phenotyping.csv"))
    print(f"[✓] Unsupervised tissue phenotyping results saved to {args.output_dir}/unsupervised_tissue_phenotyping.csv")

if __name__ == "__main__":
    main()