"""
Generate synthetic or preprocessed 10x Genomics Xenium spatial transcriptomics data.
This script creates mock files for spatial analysis pipelines.
"""

import anndata as ad 
import scanpy as sc 
import squidpy as sq
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as st
import warnings


def generate_xenium_data(output_dir="data/xenium", n_cells=1000, n_genes=2000):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Expression matrix
    expr = pd.DataFrame(
        np.random.poisson(1, size=(n_cells, n_genes)),
        columns=[f"gene{j}" for j in range(n_genes)]
    )
    expr.to_csv(f"{output_dir}/expression_matrix.csv", index_label="cell_id")
    # Spatial coordinates
    coords = pd.DataFrame({
        "cell_id": [f"cell{i}" for i in range(n_cells)],
        "x": np.random.uniform(0, 1000, n_cells),
        "y": np.random.uniform(0, 1000, n_cells)
    })
    coords.to_csv(f"{output_dir}/spatial_coords.csv", index=False)
    print(f"[✓] Synthetic Xenium data saved to {output_dir}")

def nhood_squidpy(adata, sample_key='sample', radius=50.0, cluster_key='louvain_1_1', plot_path='nhood_enrichment.png', cmap='Blues'):
    import squidpy as sq
    import matplotlib.pyplot as plt

    # Compute spatial neighbors
    sq.gr.spatial_neighbors(adata, coord_type="generic", radius=radius)
    # Compute neighborhood enrichment
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, radius=radius, key_added="nhood_enrichment")
    # Plot and save the enrichment matrix
    sq.pl.nhood_enrichment(adata, cluster_key=cluster_key, cmap=cmap)
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"[✓] Neighborhood enrichment plot saved to {plot_path}")
    return adata

if __name__ == "__main__":
    generate_xenium_data()

    # Example: Load the synthetic data as AnnData and run neighborhood enrichment
    import anndata as ad

    # Load expression matrix and coordinates
    expr = pd.read_csv("data/xenium/expression_matrix.csv", index_col=0)
    coords = pd.read_csv("data/xenium/spatial_coords.csv", index_col=None)
    adata = ad.AnnData(expr)
    adata.obs["x"] = coords["x"].values
    adata.obs["y"] = coords["y"].values
    adata.obsm["spatial"] = coords[["x", "y"]].values

    # For demonstration, create a mock cluster key and sample key
    adata.obs["louvain_1_1"] = np.random.choice(["A", "B", "C"], size=adata.n_obs)
    adata.obs["sample"] = np.random.choice(["sample1", "sample2"], size=adata.n_obs)

    # Run neighborhood enrichment and save plot
    adata = nhood_squidpy(
        adata,
        sample_key='sample',
        radius=50.0,
        cluster_key='louvain_1_1',
        plot_path="data/xenium/nhood_enrichment.png",
        cmap='Blues'
    )