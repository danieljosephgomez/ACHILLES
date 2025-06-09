# data/preprocess_and_dataloader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import os

import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf

def preprocess_expression_matrix(input_csv, output_csv, n_pcs=50):
    df = pd.read_csv(input_csv)
    gene_cols = [col for col in df.columns if col.startswith("Gene_")]
    spatial_cols = ['X_coord', 'Y_coord']

    # Normalize (Z-score)
    scaler = StandardScaler()
    norm_data = scaler.fit_transform(df[gene_cols])

    # PCA
    pca = PCA(n_components=n_pcs)
    pca_result = pca.fit_transform(norm_data)

    # UMAP
    umap_model = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    umap_result = umap_model.fit_transform(pca_result)

    # Add results to DataFrame
    for i in range(n_pcs):
        df[f"PC_{i+1}"] = pca_result[:, i]
    df['UMAP_1'] = umap_result[:, 0]
    df['UMAP_2'] = umap_result[:, 1]

    df.to_csv(output_csv, index=False)
    print(f"[✓] Preprocessed data saved to: {output_csv}")

class SpatialOmicsTorchDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.features = df[[col for col in df.columns if col.startswith("Gene_") or col.startswith("PC_")]].values.astype(np.float32)
        self.labels = df['Cluster'].values.astype(np.int64) if 'Cluster' in df.columns else np.zeros(len(df), dtype=np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_torch_dataloaders(csv_path, batch_size=64, split_ratios=(0.7, 0.2, 0.1)):
    dataset = SpatialOmicsTorchDataset(csv_path)
    total_size = len(dataset)
    train_size = int(split_ratios[0] * total_size)
    val_size = int(split_ratios[1] * total_size)
    test_size = total_size - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    return {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_set, batch_size=batch_size, shuffle=False)
    }

class SpatialOmicsTFDataset(tf.data.Dataset):
    @staticmethod
    def from_csv(csv_path, batch_size=64, shuffle=True):
        df = pd.read_csv(csv_path)
        features = df[[col for col in df.columns if col.startswith("Gene_") or col.startswith("PC_")]].values.astype(np.float32)
        labels = df['Cluster'].values.astype(np.int64) if 'Cluster' in df.columns else np.zeros(len(df), dtype=np.int64)

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(features))
        return dataset.batch(batch_size)
