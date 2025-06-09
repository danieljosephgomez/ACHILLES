import scanpy as sc
import squidpy as sq

# Load AnnData
adata = sc.read_h5ad("your_file.h5ad")

# Plot spatial cell type map
sc.pl.spatial(adata, color="cell_type", spot_size=1.2, title="Spatial Map of Cell Types")

import nico as nc

# Load data
adata = sc.read_h5ad("your_file.h5ad")

# 1. Spatial map of cell types
sc.pl.spatial(adata, color="cell_type", spot_size=1.2, title="Spatial Map of Cell Types", save="_cell_types.png")

# 2. Niche interaction map
nico_obj = nc.NiCo(adata, cell_type_key="cell_type", spatial_key="spatial")
nico_obj.fit()
nico_obj.plot_niche_map(save="niche_map.png")

# 3. Ligand-receptor interactions
nico_obj.ligand_receptor_analysis()
nico_obj.plot_ligand_receptor_network(save="ligand_receptor_network.png")

# 4. Pathway enrichment
nico_obj.pathway_enrichment()
nico_obj.plot_pathway_enrichment(save="pathway_enrichment.png")