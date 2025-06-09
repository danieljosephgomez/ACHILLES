# generate_synthetic_data.py
# Simulate 10K cells, 2K genes, 2D spatial grid
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import os

class SyntheticSpatialOmicsGenerator:
    def __init__(self, n_cells=10000, n_genes=2000, n_clusters=10, output_path="synthetic_dataset.csv", random_seed=42):
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.n_clusters = n_clusters
        self.output_path = output_path
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        self.cell_metadata = None
        self.expression_matrix = None
        self.spatial_coords = None
    
    def generate_expression_data(self):
        """
        Generate synthetic gene expression data with cluster-specific biases.
        Returns:
            expression_data: np.ndarray
            cluster_labels: np.ndarray
        """
        # Simulate gene expression with cluster-specific biases
        cluster_labels = np.random.choice(self.n_clusters, self.n_cells)
        expression_data = np.zeros((self.n_cells, self.n_genes))

        for cluster in range(self.n_clusters):
            mean_expression = np.random.rand(self.n_genes) * 2  # cluster-specific mean
            cluster_cells = (cluster_labels == cluster)
            noise = np.random.normal(0, 0.5, (np.sum(cluster_cells), self.n_genes))
            expression_data[cluster_cells] = mean_expression + noise

        expression_data = np.clip(expression_data, 0, None)  # no negative expression
        
        return expression_data, cluster_labels

        pass

    def load_czi_data(self, csv_path):
        """
        Load real CZI downloaded dataset CSV with genes and metadata.
        Expected columns: gene expression columns + 'X_coord', 'Y_coord', 'tissue', 'cell_type'
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Extract gene expression columns (assuming all except metadata columns)
        meta_cols = ['X_coord', 'Y_coord', 'tissue', 'cell_type']
        gene_cols = [col for col in df.columns if col not in meta_cols]

        self.expression_matrix = df[gene_cols].values
        self.cell_metadata = df[meta_cols]
        self.spatial_coords = df[['X_coord', 'Y_coord']].values

        self.n_cells, self.n_genes = self.expression_matrix.shape

        print(f"[✓] Loaded CZI dataset with {self.n_cells} cells and {self.n_genes} genes.")

    def save_synthetic_dataset(self, output_path):
        """
        Save current dataset (synthetic or loaded) to CSV with metadata and spatial coords.
        """
        gene_cols = [f"Gene_{i}" for i in range(self.n_genes)]
        df = pd.DataFrame(self.expression_matrix, columns=gene_cols)
        df['X_coord'] = self.spatial_coords[:, 0]
        df['Y_coord'] = self.spatial_coords[:, 1]

        # Add metadata if available
        if self.cell_metadata is not None:
            for col in self.cell_metadata.columns:
                df[col] = self.cell_metadata[col].values

        df.to_csv(output_path, index=False)
        print(f"[✓] Dataset saved to {output_path}")
    
    def generate_spatial_coordinates(self):
        # Spatially embed cells in 2D with clustering
        coords, _ = make_blobs(n_samples=self.n_cells, centers=self.n_clusters, cluster_std=5.0, random_state=self.random_seed)
        return coords

    def generate_dataset(self):
        expression_data, cluster_labels = self.generate_expression_data()
        spatial_coords = self.generate_spatial_coordinates()

        # Combine into a DataFrame
        gene_columns = [f"Gene_{i+1}" for i in range(self.n_genes)]
        df_expr = pd.DataFrame(expression_data, columns=gene_columns)
        df_expr["X_coord"] = spatial_coords[:, 0]
        df_expr["Y_coord"] = spatial_coords[:, 1]
        df_expr["Cluster"] = cluster_labels

        return df_expr
        
    @staticmethod
    def generate_synthetic_from_template(real_df, n_cells=10000):
        mean = real_df.mean().values
        std = real_df.std().values
        synthetic_data = np.random.normal(loc=mean, scale=std, size=(n_cells, len(mean)))
        synthetic_data = np.clip(synthetic_data, a_min=0, a_max=None)  # RNA counts can't be negative
        synthetic_df = pd.DataFrame(synthetic_data, columns=real_df.columns)
        
        return synthetic_df

    def save_to_csv(self):
        df = self.generate_dataset()
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"[✓] Synthetic dataset saved to: {self.output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic spatial transcriptomics data.")
    parser.add_argument("--n_cells", type=int, default=10000)
    parser.add_argument("--n_genes", type=int, default=2000)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--output", type=str, default="data/synthetic_dataset.csv")
    parser.add_argument("--random_seed", type=int, default=42) 

    args = parser.parse_args()

    generator = SyntheticSpatialOmicsGenerator(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_clusters=args.n_clusters,
        output_path=args.output
    )
    generator.save_to_csv()