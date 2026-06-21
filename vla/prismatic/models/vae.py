import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


# class VAE(nn.Module):
#     def __init__(
#           self, 
#           x_dim,
#           hidden_dim,
#           z_dim=7
#         ):
#         super(VAE, self).__init__()

#         # Define autoencoding layers
#         self.enc_layer1 = nn.Linear(x_dim, hidden_dim)
#         self.enc_layer2_mu = nn.Linear(hidden_dim, hidden_dim)
#         self.enc_layer2_logvar = nn.Linear(hidden_dim, hidden_dim)

#         # Define autoencoding layers
#         self.dec_layer1 = nn.Linear(hidden_dim, hidden_dim)
#         self.dec_layer2 = nn.Linear(hidden_dim, z_dim)

#     def encoder(self, x):
#         x = F.relu(self.enc_layer1(x))
#         mu = F.relu(self.enc_layer2_mu(x))
#         logvar = F.relu(self.enc_layer2_logvar(x))
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(logvar/2)
#         eps = torch.randn_like(std)
#         z = mu + std * eps
#         return z

#     def decoder(self, z):
#         # Define decoder network
#         output = F.relu(self.dec_layer1(z))
#         output = F.relu(self.dec_layer2(output))
#         return output

#     def forward(self, x):
#         mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         output = self.decoder(z)
#         # vae_out = (output, z, mu, logvar)
#         return output, z, mu, logvar

# Define the loss function
def vae_loss_fn(output, x, mu, logvar):
    recon_loss = F.mse_loss(output, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 3 * kl_loss


### Start of AutoEncoders

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(x.size(0), -1)  # Reshape to original
        return encoded, decoded


class VAE(AutoEncoder):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__(input_dim, hidden_dim, latent_dim)

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim, latent_dim)
        self.fc_z = nn.Linear(latent_dim, latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        encoded = self.encoder(x)

        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        z = self.reparameterize(mu, log_var)

        z = self.fc_z(z)
        decoded = self.decoder(z)
        decoded = decoded.view(x.size(0), -1)  # Reshape to original

        return encoded, decoded, mu, log_var

    def sample(self, num_samples):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            z = self.fc_z(z)
            samples = self.decoder(z)
        return samples.view(num_samples, -1)


class ConditionalVAE(VAE):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim):
        super().__init__(input_dim, hidden_dim, latent_dim)

        self.condition_dim = condition_dim

        self.y_proj = nn.Sequential(
          nn.Linear(condition_dim*latent_dim, condition_dim),  # Reduce `y` to match `projected_z`
          nn.ReLU(),
        )

        self.z_proj = nn.Sequential(
            nn.Linear(latent_dim, condition_dim),
            nn.ReLU(),
        )

        self.conditioning_proj = nn.Sequential(
            nn.Linear(condition_dim, latent_dim),
            nn.ReLU(),
        )

    def condition_on_label(self, z, y):
        y = self.y_proj(y.view(y.size(0), -1))  # Flatten conditioning input
        projected_z = self.z_proj(z)

        return self.conditioning_proj(projected_z + y)  # Ensure both have the same shape

    def forward(self, x, y):
        x = x.view(x.size(0), -1)  # Flatten input
        encoded = self.encoder(x)

        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        z = self.reparameterize(mu, log_var)

        z = self.fc_z(z)
        z = self.condition_on_label(z, y)

        decoded = self.decoder(z)
        decoded = decoded.view(x.size(0), NUM_ACTIONS_CHUNK, ACTION_DIM)  # Reshape to original

        return encoded, decoded, mu, log_var

    def sample(self, num_samples, y):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            z = self.fc_z(z)
            z = self.condition_on_label(z, y)
            samples = self.decoder(z)
        return samples.view(num_samples, -1)
