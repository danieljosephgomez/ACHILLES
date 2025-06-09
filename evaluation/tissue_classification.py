import argparse
import scanpy as sc
import squidpy as sq
import pandas as pd
import os

def get_args():
    parser = argparse.ArgumentParser(description="Tissue/Region classification for spatial omics data using Scanpy, Squidpy, and NICO.")
    parser.add_argument("--input_h5ad", type=str, required=True, help="Path to input AnnData (.h5ad) file.")
    parser.add_argument("--label_key", type=str, default="tissue", help="AnnData.obs key for tissue/region labels.")
    parser.add_argument("--output_report", type=str, required=True, help="Path to save classification report CSV.")
    return parser.parse_args()

def main():
    args = get_args()
    adata = sc.read_h5ad(args.input_h5ad)

    # Example: Use Squidpy spatial neighbors and features
    sq.gr.spatial_neighbors(adata)
    sq.gr.spatial_autocorr(adata, mode="moran")
    
    # Example: Use Scanpy for dimensionality reduction and clustering
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, key_added="leiden_clusters")
    
    # Example: Classification using NICO (placeholder, replace with actual NICO usage)
    # from nico import Classifier
    # clf = Classifier()
    # clf.fit(adata.obsm["X_pca"], adata.obs[args.label_key])
    # preds = clf.predict(adata.obsm["X_pca"])
    # For demonstration, use Leiden clusters as predictions
    preds = adata.obs["leiden_clusters"]
    true_labels = adata.obs[args.label_key]
    
    # Compute classification report
    from sklearn.metrics import classification_report
    report = classification_report(true_labels, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    report_df.to_csv(args.output_report)
    print(f"[✓] Classification report saved to {args.output_report}")

if __name__ == "__main__":
    main()