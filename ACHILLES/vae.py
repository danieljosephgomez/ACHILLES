import torch
import torch.optim as optim
from torch import nn
from torch.optim import Adam
from torchvision.models import resnet18
import torch.nn as nn
# from models.autoencoder import Autoencoder
from torchvision import models  # ✅ actively maintained
# import models

# Feature extractor (for images)
resnet = models.resnet18(pretrained=True)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # 512-dim features

input_dim = 512       # Set the input dimension here
device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 64       # Set latent dimension here

n_genes = 1000  # Limit to 1000 genes for simplicity
device = "cuda" if torch.cuda.is_available() else "cpu"

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()  # Optional, depending on normalization
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder(input_dim=n_genes, latent_dim=latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        all_embeddings = []
        for batch in data:
            x_batch = batch.to(device)
            # If you want both decoded and encoded, update forward to return both
            encoded = model.encoder(x_batch)
            _, z = model(x_batch)
            all_embeddings.append(z.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

import numpy as np
# Usage:
latent_matrix = evaluate_model(model, data)
np.savetxt("outputs/latent_embeddings.csv", latent_matrix, delimiter=",")
print("[✓] Latent embeddings saved.")
 
# Example save
np.savetxt("outputs/latent_embeddings.csv", z.cpu().detach().numpy(), delimiter=",")
