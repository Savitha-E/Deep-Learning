import numpy as np
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        latent_rep = self.encoder(x)
        output = self.decoder(latent_rep)
        return latent_rep, output


# define variables and instantiate Autoencoder class.

input_dim = 784
latent_dim = 32
hidden_dim = 128
output_dim = input_dim

my_autoencoder = AutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, output_dim=output_dim)

sample_vec = torch.rand(size=[1, input_dim])
print(sample_vec)

latent, output = my_autoencoder(sample_vec)

print(latent)
print(output)

# check how well the autoencpder worked

input_np = sample_vec.numpy()
output_np = output.detach().numpy()

print(np.std(input_np) - np.std(output_np))

