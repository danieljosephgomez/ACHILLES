"""
DGM: Deep Generative Model for single-cell and spatial omics

Features:
- Cell type proportions per spot
- Intra cell-type variation (gamma)
- Cell-type-specific gene expression imputation
- Generative modeling (sampling from latent space)
"""
import torch
from torch import nn

# Cell Type Proportions per Spot
proportions = model.get_proportions(st_adata, device=device)
st_adata.obsm["proportions"] = proportions

# Intra Cell-Type Variation (Gamma) 
gamma = model.get_gamma(st_adata, cell_type_names=cell_type_names)["B cells"]
st_adata.obsm["B_cells_gamma"] = gamma

# Cell-Type-Specific Gene Expression Imputation
indices = np.where(st_adata.obsm["proportions"][:, ct_index] > 0.03)[0]
imputed_counts = model.get_scale_for_ct("Monocyte", st_adata, indices=indices, gene_names=["Cxcl9", "Cxcl10", "Fcgr1"], cell_type_names=cell_type_names)

class DGM(nn.Module):
    """
    Deep Generative Model (VAE-style) for single-cell and spatial omics data.
    """
    def __init__(self, input_dim, latent_dim=32, hidden_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus()  # Use Softplus for count data, or Sigmoid for normalized
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
    
    def get_proportions(self, adata, batch_size=128, device="cpu"):
        """
        Returns normalized cell type proportions for each spot.
        Assumes adata.X contains the input data for each spot.
        """
        self.eval()
        loader = torch.utils.data.DataLoader(
            torch.tensor(adata.X, dtype=torch.float32), batch_size=batch_size
        )
        all_props = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                mu, logvar = self.encode(batch)
                z = self.reparameterize(mu, logvar)
                abundances = self.decode(z).cpu().numpy()  # shape: (batch, n_cell_types)
                # Normalize to get proportions per spot
                proportions = abundances / (abundances.sum(axis=1, keepdims=True) + 1e-8)
                all_props.append(proportions)
        all_props = np.vstack(all_props)
        # Optionally, return as DataFrame with cell type names if available
        return all_props
    
    def get_gamma(self, adata, batch_size=128, device="cpu", cell_type_names=None):
        """
        Returns the latent variables (gamma) for each cell type per spot.
        Assumes adata.X contains the input data for each spot.
        Optionally provide cell_type_names (list of str) for output DataFrame columns.
        """
        self.eval()
        loader = torch.utils.data.DataLoader(
            torch.tensor(adata.X, dtype=torch.float32), batch_size=batch_size
        )
        gammas = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                mu, logvar = self.encode(batch)
                # Use mu as the mean of the latent variable distribution
                gammas.append(mu.cpu().numpy())
        gammas = np.vstack(gammas)
        if cell_type_names is not None:
            return pd.DataFrame(gammas, columns=cell_type_names)
        return gammas

    def get_scale_for_ct(self, cell_type_name, adata, indices=None, gene_names=None, batch_size=128, device="cpu", cell_type_names=None):
        """
        Impute cell-type-specific gene expression for a given cell type and spots.
        - cell_type_name: str, name of the cell type (must match cell_type_names)
        - adata: AnnData object
        - indices: indices of spots to impute (default: all)
        - gene_names: list of gene names to return (default: all)
        - cell_type_names: list of all cell type names (order must match decoder output)
        Returns a DataFrame: rows=spots, columns=genes
        """
        self.eval()
        if indices is None:
            indices = np.arange(adata.n_obs)
        X = adata.X[indices]
        loader = torch.utils.data.DataLoader(
            torch.tensor(X, dtype=torch.float32), batch_size=batch_size
        )
        ct_idx = cell_type_names.index(cell_type_name) if cell_type_names else 0
        imputed = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                mu, logvar = self.encode(batch)
                z = self.reparameterize(mu, logvar)
                decoded = self.decode(z).cpu().numpy()  # shape: (batch, n_cell_types or n_genes)
                # If decoder output is (batch, n_cell_types, n_genes), select cell type
                if decoded.ndim == 3:
                    decoded = decoded[:, ct_idx, :]
                imputed.append(decoded)
        imputed = np.vstack(imputed)
        if gene_names is not None:
            return pd.DataFrame(imputed, columns=gene_names, index=indices)[gene_names]
        return pd.DataFrame(imputed, index=indices)

# Example usage after training:
# proportions = model.get_proportions(st_adata, device=device)
# st_adata.obsm["proportions"] = proportions

# To plot a heatmap for a cell type (e.g., "B cells"):
# import scanpy as sc
# st_adata.obs['B cells'] = st_adata.obsm['proportions'][:, b_cell_index]  # replace b_cell_index with correct column
# sc.pl.spatial(st_adata, color="B cells", spot_size=130)