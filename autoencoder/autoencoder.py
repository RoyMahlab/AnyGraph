import torch.nn as nn
import torch.nn.functional as F


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        layers_sizes = [input_dim*2, input_dim*4, input_dim*2]
        self.fc1 = nn.Linear(input_dim, layers_sizes[0])
        self.bn1 = nn.BatchNorm1d(layers_sizes[0])
        self.fc2 = nn.Linear(layers_sizes[0], layers_sizes[1])
        self.bn2 = nn.BatchNorm1d(layers_sizes[1])
        self.fc3 = nn.Linear(layers_sizes[1], layers_sizes[2])
        self.bn3 = nn.BatchNorm1d(layers_sizes[2])
        self.fc4 = nn.Linear(layers_sizes[2], latent_dim)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))), p=0.1, training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))), p=0.1, training=self.training)
        x = F.dropout(F.relu(self.bn3(self.fc3(x))), p=0.1, training=self.training)
        return self.fc4(x)


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        layers_sizes = [latent_dim*2, latent_dim*4, latent_dim*2]
        self.fc1 = nn.Linear(latent_dim, layers_sizes[0])
        self.bn1 = nn.BatchNorm1d(layers_sizes[0])
        self.fc2 = nn.Linear(layers_sizes[0], layers_sizes[1])
        self.bn2 = nn.BatchNorm1d(layers_sizes[1])
        self.fc3 = nn.Linear(layers_sizes[1], layers_sizes[2])
        self.bn3 = nn.BatchNorm1d(layers_sizes[2])
        self.fc4 = nn.Linear(layers_sizes[2], latent_dim)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))), p=0.1, training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))), p=0.1, training=self.training)
        x = F.dropout(F.relu(self.bn3(self.fc3(x))), p=0.1, training=self.training)
        return self.fc4(x)


# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return latent, recon
