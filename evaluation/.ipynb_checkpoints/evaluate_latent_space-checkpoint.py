import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import os

# --- Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description="Evaluate latent space from spatial omics foundation model.")
    parser.add_argument("--input_csv", type=str, default="outputs/latent_embeddings.csv", help="Path to latent embeddings CSV.")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis", help="Directory to save plots.")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters for KMeans.")
    parser.add_argument("--umap_neighbors", type=int, default=15, help="UMAP number of neighbors.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP minimum distance.")
    return parser.parse_args()

# --- Main Function ---
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Data ---
    print("[INFO] Loading latent embeddings...")
    df = pd.read_csv(args.input_csv)
    latent_cols = [col for col in df.columns if col.startswith("Latent_")]
    coords_cols = ["X_coord", "Y_coord"]
    latent = df[latent_cols].values
    spatial = df[coords_cols].values

    # --- UMAP ---
    print("[INFO] Running UMAP...")
    reducer = umap.UMAP(n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist, random_state=42)
    umap_coords = reducer.fit_transform(latent)
    df["UMAP1"] = umap_coords[:, 0]
    df["UMAP2"] = umap_coords[:, 1]

    # --- Clustering ---
    print(f"[INFO] Clustering into {args.n_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(latent)

    # --- Evaluation ---
    print("[INFO] Calculating silhouette score...")
    sil_score = silhouette_score(latent, df["Cluster"])
    print(f"[✓] Silhouette Score: {sil_score:.4f}")

    # --- Plot UMAP ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="UMAP1", y="UMAP2", hue="Cluster", palette="tab10", data=df, s=10)
    plt.title("UMAP Projection of Latent Space")
    plt.savefig(f"{args.output_dir}/umap_clusters.png", dpi=300)
    plt.close()

    # --- Plot Spatial Map ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="X_coord", y="Y_coord", hue="Cluster", palette="tab10", data=df, s=10)
    plt.title("Spatial Distribution of Clusters")
    plt.savefig(f"{args.output_dir}/spatial_clusters.png", dpi=300)
    plt.close()

    # --- Save Labeled Data ---
    df.to_csv(f"{args.output_dir}/latent_with_umap_clusters.csv", index=False)
    print(f"[✓] Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
