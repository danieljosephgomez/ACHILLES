"""
Process and analyze 10x Genomics cell-vdj (immune profiling) data using Scirpy.
Loads filtered_feature_bc_matrix.h5 and filtered_contig_annotations.csv from 10x Genomics.
Performs TCR/BCR analysis and visualizations using Scirpy's IR model.
See:
- https://scirpy.scverse.org/en/latest/data-structure.html
- https://scirpy.scverse.org/en/latest/ir-biology.html
- https://scirpy.scverse.org/en/latest/tutorials/tutorial_io.html
- https://scirpy.scverse.org/en/latest/tutorials/tutorial_3k_tcr.html
"""

import scirpy as ir
import scanpy as sc
from pathlib import Path

def load_and_integrate(expr_path, vdj_path):
    if not Path(expr_path).exists():
        print(f"[ERROR] Expression file not found: {expr_path}")
        return None
    if not Path(vdj_path).exists():
        print(f"[ERROR] VDJ annotation file not found: {vdj_path}")
        return None

    print(f"[INFO] Loading gene expression data from {expr_path}")
    adata = sc.read_10x_h5(expr_path)
    print(f"[INFO] AnnData shape: {adata.shape}")

    print(f"[INFO] Loading VDJ data from {vdj_path}")
    ir.io.read_10x_vdj(vdj_path, adata)
    print(f"[INFO] VDJ data loaded and integrated.")
    return adata

def basic_ir_analysis(adata):
    # Chain QC and clonotype definition
    ir.tl.chain_qc(adata)
    ir.tl.define_clonotypes(adata)
    print("[INFO] IR chain QC and clonotype definition complete.")

def plot_ir(adata):
    # UMAP colored by clonotype (if UMAP exists)
    if "X_umap" in adata.obsm:
        ir.pl.umap(adata, color="ir_clonotype", save="_clonotype.png", show=False)
    # Spectratype plot
    ir.pl.spectratype(adata, groupby=None, save="_spectratype.png", show=False)
    # Clonotype network
    ir.pl.clonotype_network(adata, color="ir_clonotype", save="_clonotype_network.png", show=False)
    # Clonotype abundance
    ir.pl.clonotype_abundance(adata, groupby=None, target_col="clonotype", save="_clonotype_abundance.png", show=False)
    print("[INFO] IR plots generated (see figures/ directory).")

if __name__ == "__main__":
    expr_path = "data/10k_Human_PBMC_5p_v3_filtered_feature_bc_matrix.h5"
    vdj_path = "data/filtered_contig_annotations.csv"
    adata = load_and_integrate(expr_path, vdj_path)
    if adata is not None:
        basic_ir_analysis(adata)
        plot_ir(adata)