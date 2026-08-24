import torch
import torch.nn as nn
import numpy as np
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=32):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Encode the input
        latent = self.encoder(x)
        # Decode the latent representation
        reconstructed = self.decoder(latent)
        return latent, reconstructed
def main():


    # Instantiate the autoencoder
    input_dim = 784
    hidden_dim = 128
    latent_dim = 32
    autoencoder = Autoencoder(input_dim, hidden_dim, latent_dim)

    # Generate a random 784-dimensional input sample
    X = torch.randn(1, input_dim)

    # Get the latent representation
    latent_representation, r = autoencoder(X)

    print("Latent Representation:", latent_representation)
    print(np.std(X)[0]-np.std(r)[0])


if __name__ == "__main__":
    main()
