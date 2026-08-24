import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

# define hyperparameters:

latent_dim = 10  # Dimension of the latent vector (input to the generator)
hidden_dim = 128  # Hidden layer size
output_dim = 1  # Output dimension (1D data point for simplicity)
batch_size = 64
num_epochs = 5000
learning_rate = 0.0002


# define discriminator:

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=output_dim, out_features=hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=hidden_dim, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output


# define gen

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        )

    def forward(self, x):
        output = self.model(x)
        return output


generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()
optimizer_D = Adam(params=discriminator.parameters(), lr=learning_rate)
optimizer_G = Adam(params=generator.parameters(), lr=learning_rate)


for epoch in range(num_epochs):

    # generate real data
    real_data = torch.normal(mean=0, std=1, size=(batch_size, output_dim))
    real_labels = torch.ones(size=(batch_size, 1))

    # generate fake data
    latent_samples = torch.randn(size=(batch_size, latent_dim))
    fake_data = generator(latent_samples)
    fake_labels = torch.zeros(size=(batch_size, 1))

    # all data:
    all_samples = torch.cat((real_data, fake_data))
    all_labels = torch.cat((real_labels, fake_labels))

    # train discriminator:
    optimizer_D.zero_grad()
    output_discriminator = discriminator(all_samples)
    loss_discriminator = criterion(output_discriminator, all_labels)
    loss_discriminator.backward()
    optimizer_D.step()

    # train generator
    optimizer_G.zero_grad()
    latent_samples = torch.randn(size=(batch_size, latent_dim))
    generated_samples = generator(latent_samples)
    disc_output_generated = discriminator(generated_samples)
    loss_generator = criterion(disc_output_generated, real_labels)
    loss_generator.backward()
    optimizer_G.step()

    # print losses
    if epoch % 10 == 0:
        print(f"Epoch: {epoch + 1}")
        print(f"Loss Discriminator: {loss_discriminator}")
        print(f"Loss Generator: {loss_generator}")

# plot real vs fake data

real_data = torch.normal(size=(batch_size, output_dim), mean=0, std=1).numpy()
latent_space = torch.randn(size=(batch_size, latent_dim))
generated_data = generator(latent_space).detach().numpy()

plt.hist(real_data, bins=30, label="Real Data")
plt.hist(generated_data, bins=30, label="Generated Data")
plt.show()


