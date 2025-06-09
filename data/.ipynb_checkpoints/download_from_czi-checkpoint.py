import cellxgene_census
import pandas as pd
import numpy as np
import argparse
import os
import cellxgene_census

tissues = [
    "brain",
    "cortex",
    "hippocampus",
    "heart",
    "lung",
    "intestine",
    "small intestine",
    "colon",
    "muscle",
    "bone marrow",
    "pbmc",
    "lymph node",
    "tonsil"
]

with cellxgene_census.open_soma() as census:
    human = census["census_data"]["homo_sapiens"]
    tissues = human.obs.read(
        value_filter="is_primary_data == True",
        column_names=["tissue_general"]
    ).to_pandas()["tissue_general"].unique()

    # Sample 10k cells across all organs/tissues
    obs_df = human.obs.read(value_filter="is_primary_data == True").to_pandas()
    sampled_obs = obs_df.sample(n=10000, random_state=42)

    # Select expression data for those cells
    data = human.X["raw"].read(coords=(sampled_obs.index, slice(None))).to_numpy()

    # Save gene matrix
    genes = human.var.to_pandas().index
    df = pd.DataFrame(data, columns=genes, index=sampled_obs.index)
    df.to_csv("data/czi_sampled_matrix.csv")
    print("[✓] Saved real-world gene expression matrix")

def download_czi_sample(output_path: str, n_cells_per_tissue: int = 2000, seed: int = 42):
    print(f"[INFO] Downloading {n_cells} human primary cells from CZI CELLxGENE Census...")
    
    with cellxgene_census.open_soma() as census:
        human = census["census_data"]["homo_sapiens"]

        dfs = []
        np.random.seed(seed)

        for tissue in tissues:
            print(f"[INFO] Sampling {n_cells_per_tissue} cells from tissue: {tissue}")
            # Adjust filter for lymph-node special case if needed:
            filter_str = f"is_primary_data == True and tissue_general == '{tissue}'"

            # Read cell metadata (obs table)
            obs_df = human.obs.read(
                column_names=["soma_joinid", "tissue_general", "cell_type"]
            ).to_pandas()

            if len(obs_df) == 0:
                print(f"[WARN] No cells found for tissue '{tissue}'. Skipping.")
                continue

            # Sample n_cells_per_tissue cells, or all if fewer available
            sample_size = min(n_cells_per_tissue, len(obs_df))
            sampled_obs = obs_df.sample(n=sample_size, random_state=seed).sort_values("soma_joinid")
            sampled_obs.reset_index(drop=True, inplace=True)

            # Fetch expression matrix for sampled cells
            X = human.X["raw"].read(
                coords=(sampled_obs["soma_joinid"].values, slice(None))
            ).to_numpy()

            genes = human.var.read(column_names=["feature_id"]).to_pandas()["feature_id"].values

            # Create DataFrame for this tissue batch
            df = pd.DataFrame(X, columns=[f"Gene_{g}" for g in genes])
            df["X_coord"] = np.random.uniform(0, 1000, size=sample_size)
            df["Y_coord"] = np.random.uniform(0, 1000, size=sample_size)
            df["tissue"] = sampled_obs["tissue_general"].values
            df["cell_type"] = sampled_obs["cell_type"].values

            dfs.append(df)

            # Concatenate all tissues into one DataFrame
            full_df = pd.concat(dfs, ignore_index=True)
    
            full_df.to_csv(output_path, index=False)
            print(f"[✓] Saved combined multi-tissue dataset to {output_path}")
            
            
            print("[INFO] Fetching expression matrix...")
            X = human.X["raw"].read(
                coords=(sampled_obs["soma_joinid"].values, slice(None))
            ).to_numpy()

            var_df = human.var.read(column_names=["feature_id"]).to_pandas()
            genes = var_df["feature_id"].values

            print("[INFO] Creating synthetic spatial coordinates...")
            # Create fake 2D spatial grid
            coords = np.random.uniform(0, 1000, size=(n_cells, 2))
    
            # Save as DataFrame
            df = pd.DataFrame(X, columns=[f"Gene_{g}" for g in genes])
            df["X_coord"] = coords[:, 0]
            df["Y_coord"] = coords[:, 1]
    
            # Optionally add metadata
            df["tissue"] = sampled_obs["tissue_general"].values
            df["cell_type"] = sampled_obs["cell_type"].values
    
            # Save to CSV
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"[✓] Saved dataset to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download real cell x gene matrix from CZI CELLxGENE.")
    parser.add_argument("--output_path", type=str, default="data/czi_sampled_matrix.csv")
    parser.add_argument("--n_cells", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    download_czi_sample(args.output_path, args.n_cells, args.seed)
