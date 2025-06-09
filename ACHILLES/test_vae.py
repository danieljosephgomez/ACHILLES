import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# Add root path to Python path
sys.path.append(os.path.abspath(".."))

# Import your Autoencoder model
from models.autoencoder import Autoencoder

# Simulated input dimensions
input_dim = 512       # Output from ResNet feature extractor
latent_dim = 64

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model, criterion, optimizer
model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Simulate test_loader with random data
# This emulates ResNet18 512-d features from 100 samples
dummy_data = torch.randn(100, input_dim)
test_loader = torch.utils.data.DataLoader(dummy_data, batch_size=16)

# Run inference to extract latent embeddings
model.eval()
all_embeddings = []

with torch.no_grad():
    for batch in test_loader:
        x_batch = batch.to(device)
        _, z = model(x_batch)
        all_embeddings.append(z.cpu().numpy())

# Concatenate and save
latent_matrix = np.concatenate(all_embeddings, axis=0)
os.makedirs("outputs", exist_ok=True)
np.savetxt("outputs/latent_embeddings.csv", latent_matrix, delimiter=",")
print("[✓] Latent embeddings saved to outputs/latent_embeddings.csv")
