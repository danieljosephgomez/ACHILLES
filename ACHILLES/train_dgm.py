# This script trains an autoencoder on a 10x Genomics dataset and extracts latent embeddings
from models.dgm import DGM
import argparse
import pandas as pd
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models.autoencoder import Autoencoder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

n_genes = genes.shape[1]
model = DGM(input_dim=n_genes, latent_dim=32).to(DEVICE)

# --- Config ---
DATA_PATH = "data/10k_Human_PBMC_5p_v3_filtered_feature_bc_matrix.h5"  # Example 10x file
MODEL_SAVE_PATH = "models/foundation_autoencoder.pt"
LATENT_EMBEDDING_PATH = "models/latent_embeddings.csv"
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load & Normalize ---
print("[INFO] Loading 10x Genomics dataset...")
adata = sc.read_10x_h5(DATA_PATH)
# Optionally, filter genes/cells as needed
genes = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
scaler = StandardScaler()
genes_scaled = scaler.fit_transform(genes.astype(np.float32))

# --- TensorLoader ---
tensor_data = torch.tensor(genes_scaled, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model ---
n_genes = genes.shape[1]
model = Autoencoder(input_dim=n_genes, latent_dim=32).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
print("[INFO] Starting training...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in loader:
        x = batch[0].to(DEVICE)
        optimizer.zero_grad()
        x_hat = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f}")

# --- Save Model ---
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"[✓] Model saved to {MODEL_SAVE_PATH}")

# --- Extract and Save Latent Embeddings ---
print("[INFO] Extracting latent embeddings...")
model.eval()
with torch.no_grad():
    full_tensor = torch.tensor(genes_scaled, dtype=torch.float32).to(DEVICE)
    latent = model.encoder(full_tensor).cpu().numpy()
    
latent_df = pd.DataFrame(latent, columns=[f"Latent_{i+1}" for i in range(latent.shape[1])])
if args.data_type == "single-cell" and "spatial" in adata.obsm:
    latent_df["X_coord"] = adata.obsm["single-cell"][:, 0]
    latent_df["Y_coord"] = adata.obsm["spatial"][:, 1]
latent_df["Barcode"] = adata.obs_names

# Save
os.makedirs(os.path.dirname(LATENT_EMBEDDING_PATH), exist_ok=True)
latent_df.to_csv(LATENT_EMBEDDING_PATH, index=False)
print(f"[✓] Latent embeddings saved to {LATENT_EMBEDDING_PATH}")