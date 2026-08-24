import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
latent_dim = 10
hidden_dim = 128
output_dim = 1
batch_size = 64
num_epochs = 5000
learning_rate = 0.0002


# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def main():
    # Initialize models
    generator = Generator()
    discriminator = Discriminator()

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Training
    for epoch in range(num_epochs):
        # Generate real data (1D Gaussian distribution)
        real_data = torch.normal(mean=0, std=1, size=(batch_size, output_dim))
        real_labels = torch.ones(batch_size, 1)

        # Generate fake data
        latent_vectors = torch.randn(batch_size, latent_dim)
        fake_data = generator(latent_vectors)
        fake_labels = torch.zeros(batch_size, 1)

        # Train Generator
        optimizer_G.zero_grad()
        generated_data = generator(latent_vectors)
        generator_preds = discriminator(generated_data)
        g_loss = criterion(generator_preds, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_preds = discriminator(real_data)
        real_loss = criterion(real_preds, real_labels)
        fake_preds = discriminator(fake_data.detach())
        fake_loss = criterion(fake_preds, fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Print losses occasionally
        if epoch % 500 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Plot real vs. generated data
    real_samples = torch.normal(mean=0, std=1, size=(1000, 1)).numpy()
    generated_samples = generator(torch.randn(1000, latent_dim)).detach().numpy()

    plt.figure(figsize=(10, 5))
    plt.hist(real_samples, bins=30, alpha=0.6, label="Real Data")
    plt.hist(generated_samples, bins=30, alpha=0.6, label="Generated Data")
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Real vs. Generated Data Distribution")
    plt.show()


if __name__ == "__main__":
    main()
