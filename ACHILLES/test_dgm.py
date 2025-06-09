import torch
import numpy as np
from dgm import DGM

def test_dgm_forward():
    input_dim = 100
    latent_dim = 8
    batch_size = 16
    model = DGM(input_dim=input_dim, latent_dim=latent_dim)
    x = torch.randn(batch_size, input_dim)
    recon_x, mu, logvar, z = model(x)
    assert recon_x.shape == (batch_size, input_dim), "Reconstruction shape mismatch"
    assert mu.shape == (batch_size, latent_dim), "Mu shape mismatch"
    assert logvar.shape == (batch_size, latent_dim), "Logvar shape mismatch"
    assert z.shape == (batch_size, latent_dim), "Latent z shape mismatch"
    print("test_dgm_forward passed.")

def test_get_proportions():
    input_dim = 50
    latent_dim = 4
    n_spots = 10
    model = DGM(input_dim=input_dim, latent_dim=latent_dim)
    # Fake AnnData-like object
    class DummyAdata:
        X = np.random.rand(n_spots, input_dim)
    adata = DummyAdata()
    props = model.get_proportions(adata)
    assert props.shape == (n_spots, input_dim), "Proportions shape mismatch"
    print("test_get_proportions passed.")

if __name__ == "__main__":
    test_dgm_forward()
    test_get_proportions()