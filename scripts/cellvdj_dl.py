"""
Download 10x Genomics cell-vdj (immune profiling) datasets.

This script downloads selected cell-vdj datasets from 10x Genomics public resources.
You can specify the dataset URL and output path, or use a preset.

Example usage:
    python cell_vdj_dl.py --preset pbmc
    python cell_vdj_dl.py --url https://cf.10xgenomics.com/samples/cell-vdj/9.0.1/10k_Human_PBMC_5p_v3_Ultima_10k_Human_PBMC_5p_v3_Ultima/10k_Human_PBMC_5p_v3_Ultima_10k_Human_PBMC_5p_v3_Ultima_count_sample_filtered_feature_bc_matrix.h5 --output data/10k_Human_PBMC_5p_v3_filtered_feature_bc_matrix.h5
"""

import argparse
import requests
from pathlib import Path

PRESETS = {
    "pbmc": {
        "url": "https://cf.10xgenomics.com/samples/cell-vdj/9.0.1/10k_Human_PBMC_5p_v3_Ultima_10k_Human_PBMC_5p_v3_Ultima/10k_Human_PBMC_5p_v3_Ultima_10k_Human_PBMC_5p_v3_Ultima_count_sample_filtered_feature_bc_matrix.h5",
        "output": "data/10k_Human_PBMC_5p_v3_filtered_feature_bc_matrix.h5"
    }
}

def download_file(url, output_path, chunk_size=8192):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    print(f"[✓] Downloaded: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 10x Genomics cell-vdj dataset.")
    parser.add_argument("--preset", type=str, choices=PRESETS.keys(), help="Preset dataset to download (pbmc).")
    parser.add_argument("--url", type=str, help="Direct download URL from 10x Genomics.")
    parser.add_argument("--output", type=str, help="Output file path (required if --url is used).")
    args = parser.parse_args()

    if args.preset:
        preset = PRESETS[args.preset]
        download_file(preset["url"], preset["output"])
    elif args.url and args.output:
        download_file(args.url, args.output)
    else:
        print("Specify either --preset or both --url and --output.")