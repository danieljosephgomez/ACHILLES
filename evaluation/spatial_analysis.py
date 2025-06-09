import argparse
import scanpy as sc
import squidpy as sq
import os

def get_args():
    parser = argparse.ArgumentParser(description="Spatial analysis for spatial omics data using Squidpy.")
    parser.add_argument("--input_h5ad", type=str, required=True, help="Path to input AnnData (.h5ad) file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results and plots.")
    parser.add_argument("--label_key", type=str, default="cell_type", help="AnnData.obs key for cell/tissue labels.")
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    adata = sc.read_h5ad(args.input_h5ad)

    # --- 1. Compute spatial neighbors ---
    print("[INFO] Computing spatial neighbors...")
    sq.gr.spatial_neighbors(adata)

    # --- 2. Spatial autocorrelation (Moran's I) ---
    print("[INFO] Calculating spatial autocorrelation (Moran's I)...")
    sq.gr.spatial_autocorr(adata, mode="moran")
    moran_df = adata.uns["moranI"]
    moran_df.to_csv(os.path.join(args.output_dir, "spatially_variable_genes.csv"))
    print(f"[✓] Moran's I results saved to {args.output_dir}/spatially_variable_genes.csv")

    # --- 3. Ligand-Receptor interaction analysis ---
    print("[INFO] Running ligand-receptor interaction analysis...")
    sq.gr.ligrec(adata, cluster_key=args.label_key)
    ligrec_df = sq.gr.ligrec_results_to_df(adata)
    ligrec_df.to_csv(os.path.join(args.output_dir, "ligand_receptor_interactions.csv"), index=False)
    print(f"[✓] Ligand-receptor interactions saved to {args.output_dir}/ligand_receptor_interactions.csv")

    # --- 4. Plot spatial scatter of cell types ---
    print("[INFO] Plotting spatial scatter of cell types...")
    sc.pl.spatial(adata, color=args.label_key, save=f"_{args.label_key}_spatial.png", show=False)
    # Move the plot to output_dir if needed
    plot_name = f"spatial_{args.label_key}_spatial.png"
    if os.path.exists(plot_name):
        os.replace(plot_name, os.path.join(args.output_dir, plot_name))

    # --- 5. Neighborhood enrichment analysis ---
    print("[INFO] Running neighborhood enrichment analysis...")
    sq.gr.nhood_enrichment(adata, cluster_key=args.label_key)
    sq.pl.nhood_enrichment(adata, cluster_key=args.label_key, save="_nhood_enrichment.png", show=False)
    # Move the plot to output_dir if needed
    plot_name = "spatial_nhood_enrichment.png"
    if os.path.exists(plot_name):
        os.replace(plot_name, os.path.join(args.output_dir, plot_name))

    print(f"[✓] Spatial analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()