import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from pathlib import Path

def preprocess_expression_matrix(input_csv, output_csv=None, n_pcs=50, plot_prefix="plots"):
    df = pd.read_csv(input_csv)
    gene_cols = [col for col in df.columns if col.startswith("Gene_")]
    spatial_cols = ['X_coord', 'Y_coord']

    # Normalize (Z-score)
    scaler = StandardScaler()
    norm_data = scaler.fit_transform(df[gene_cols])

    # PCA
    plt.figure(figsize=(8,6))
    pca = PCA(n_components=n_pcs)
    pca_result = pca.fit_transform(norm_data)
    Path(plot_prefix).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{plot_prefix}/pca_plot.png")
    plt.close()

    # UMAP
    plt.figure(figsize=(8,6))
    umap_model = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    umap_result = umap_model.fit_transform(pca_result)
    plt.savefig(f"{plot_prefix}/umap_plot.png")
    plt.close()

    # Add results to DataFrame
    for i in range(n_pcs):
        df[f"PC_{i+1}"] = pca_result[:, i]
    df['UMAP_1'] = umap_result[:, 0]
    df['UMAP_2'] = umap_result[:, 1]

    # Plot PCA (first two components)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=df['Cluster'] if 'Cluster' in df.columns else None, palette='tab10', s=10)
    plt.title("PCA: PC1 vs PC2")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left') if 'Cluster' in df.columns else None
    plt.tight_layout()
    Path(plot_prefix).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{plot_prefix}/pca_plot.png")
    plt.close()

    # Plot UMAP
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=umap_result[:,0], y=umap_result[:,1], hue=df['Cluster'] if 'Cluster' in df.columns else None, palette='tab10', s=10)
    plt.title("UMAP")
    plt.xlabel("UMAP_1")
    plt.ylabel("UMAP_2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left') if 'Cluster' in df.columns else None
    plt.tight_layout()
    plt.savefig(f"{plot_prefix}/umap_plot.png")
    plt.close()

    print(f"[✓] PCA and UMAP plots saved to {plot_prefix}/")

    # Optionally save CSV
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"[✓] Preprocessed data saved to: {output_csv}")

# Update the CLI to allow skipping CSV and specifying plot directory
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Preprocess spatial omics CSV with PCA and UMAP, and generate plots."
    )
    parser.add_argument(
        "--input", type=str, default="data/synthetic_dataset.csv",
        help="Path to input CSV file (default: data/synthetic_dataset.csv)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save preprocessed CSV (default: None, only plots will be generated)"
    )
    parser.add_argument(
        "--n_pcs", type=int, default=50,
        help="Number of principal components (default: 50)"
    )
    parser.add_argument(
        "--plot_prefix", type=str, default="plots",
        help="Directory prefix for saving plots (default: plots)"
    )
    args = parser.parse_args()
    preprocess_expression_matrix(args.input, args.output, args.n_pcs, args.plot_prefix)