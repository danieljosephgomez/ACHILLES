"""
Patient stratification for spatial omics with single-cell precision,
tailored for cancer immunotherapy and immuno-oncology contexts.

This script aggregates spatial omics features (e.g., cell type composition,
immune cell infiltration, spatial metrics) per patient and clusters patients
to identify subgroups relevant for therapy response and prognosis.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import os

def get_args():
    parser = argparse.ArgumentParser(description="Spatial omics patient stratification for immuno-oncology.")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV with single-cell spatial features and patient IDs.")
    parser.add_argument("--patient_col", type=str, default="patient_id", help="Column name for patient IDs.")
    parser.add_argument("--celltype_col", type=str, default="cell_type", help="Column name for cell type annotations.")
    parser.add_argument("--output_report", type=str, required=True, help="Path to save patient stratification CSV.")
    parser.add_argument("--n_groups", type=int, default=4, help="Number of patient groups to stratify.")
    return parser.parse_args()

def aggregate_patient_features(df, patient_col, celltype_col):
    # Example: immune cell fraction, T cell density, spatial diversity, etc.
    celltype_counts = df.groupby([patient_col, celltype_col]).size().unstack(fill_value=0)
    celltype_frac = celltype_counts.div(celltype_counts.sum(axis=1), axis=0)
    # Example: spatial diversity (Shannon entropy)
    shannon_entropy = -np.nansum(celltype_frac * np.log(celltype_frac + 1e-9), axis=1)
    features = celltype_frac.copy()
    features["shannon_entropy"] = shannon_entropy
    features.index.name = patient_col
    return features

def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    df = pd.read_csv(args.input_csv)

    # Aggregate features per patient
    features = aggregate_patient_features(df, args.patient_col, args.celltype_col)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Cluster patients
    clustering = AgglomerativeClustering(n_clusters=args.n_groups)
    patient_groups = clustering.fit_predict(X)
    features["stratification_group"] = patient_groups

    # Evaluate clustering
    sil_score = silhouette_score(X, patient_groups)
    print(f"[✓] Silhouette score for patient stratification: {sil_score:.3f}")

    # Save results
    features.reset_index().to_csv(args.output_report, index=False)
    print(f"[✓] Patient stratification report saved to {args.output_report}")

if __name__ == "__main__":
    main()