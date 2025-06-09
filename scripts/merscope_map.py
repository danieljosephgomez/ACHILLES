"""
Generate synthetic or preprocessed Vizgen MERSCOPE spatial transcriptomics data.
This script creates mock cell_by_gene.csv and cell_metadata.csv files.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_merscope_data(output_dir="data/synthetic_merscope", n_cells=1000, n_genes=2000):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # cell_by_gene.csv
    cell_by_gene = pd.DataFrame(
        np.random.poisson(1, size=(n_cells, n_genes)),
        columns=[f"gene{j}" for j in range(n_genes)]
    )
    cell_by_gene.insert(0, "cell_id", [f"cell{i}" for i in range(n_cells)])
    cell_by_gene.to_csv(f"{output_dir}/cell_by_gene.csv", index=False)
    # cell_metadata.csv
    cell_metadata = pd.DataFrame({
        "cell_id": [f"cell{i}" for i in range(n_cells)],
        "x": np.random.uniform(0, 1000, n_cells),
        "y": np.random.uniform(0, 1000, n_cells),
        "cell_type": np.random.choice(["T", "B", "Myeloid", "Stromal"], n_cells)
    })
    cell_metadata.to_csv(f"{output_dir}/cell_metadata.csv", index=False)
    print(f"[✓] Synthetic MERSCOPE data saved to {output_dir}")

if __name__ == "__main__":
    generate_merscope_data()