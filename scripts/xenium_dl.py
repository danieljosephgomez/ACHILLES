"""
Download 10x Genomics single-cell and spatial transcriptomics datasets.

This script downloads selected datasets from 10x Genomics public resources.
You can specify the dataset URL and output path.

Example usage:
    python download_10x.py --url https://cf.10xgenomics.com/samples/cell-vdj/9.0.1/10k_Human_PBMC_5p_v3_Ultima_10k_Human_PBMC_5p_v3_Ultima/10k_Human_PBMC_5p_v3_Ultima_10k_Human_PBMC_5p_v3_Ultima_count_sample_filtered_feature_bc_matrix.h5 --output data/10k_Human_PBMC_5p_v3_filtered_feature_bc_matrix.h5
    python download_10x.py --url https://cf.10xgenomics.com/samples/xenium/1.3.0/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.zip --output data/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.zip
    python download_10x.py --url https://cf.10xgenomics.com/samples/xenium/1.3.0/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs.zip --output data/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs.zip
    python download_10x.py --url https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hKidney_nondiseased_section/Xenium_V1_hKidney_nondiseased_section_outs.zip --output data/Xenium_V1_hKidney_nondiseased_section_outs.zip
"""

import argparse
import requests
from pathlib import Path

PRESETS = {
    "brain": {
        "url": "https://cf.10xgenomics.com/samples/xenium/1.3.0/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs.zip",
        "output": "data/Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs.zip"
    },
    "lung": {
        "url": "https://cf.10xgenomics.com/samples/xenium/1.3.0/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.zip",
        "output": "data/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.zip"
    },
    "kidney": {
        "url": "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hKidney_nondiseased_section/Xenium_V1_hKidney_nondiseased_section_outs.zip",
        "output": "data/Xenium_V1_hKidney_nondiseased_section_outs.zip"
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
    parser = argparse.ArgumentParser(description="Download 10x Genomics single-cell or spatial transcriptomics dataset.")
    parser.add_argument("--preset", type=str, choices=PRESETS.keys(), help="Preset dataset to download (brain, lung, kidney).")
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