import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from autoencoder import Autoencoder
from data_loader import get_dataloaders
from hyperparameters import get_hyperparameters
from typing import List, Dict

def multi_dataset_cosine_similarity_loss(latents: List):
    latents = [latent / latent.norm(dim=0, keepdim=True) + 1e-8 for latent in latents]
    same_dataset_cosine_losses = []
    between_dataset_cosine_losses = []
    for i in range(len(latents)):
        latent_i = latents[i]
        cosine_matrix = torch.mm(latent_i.t(), latent_i)
        diagonal = torch.eye(cosine_matrix.shape[0], device=latent_i.device)
        same_dataset_cosine_losses.append((cosine_matrix - diagonal).mean()) # Penalize similarity between datasets
        for j in range(i + 1, len(latents)):
            latent_j = latents[j]
            cosine_matrix = torch.mm(latent_i.t(), latent_j)
            between_dataset_cosine_losses.append(1 - cosine_matrix.mean())  # Penalize disimilarity between datasets
    between_dataset_cosine_loss = sum(between_dataset_cosine_losses) / len(between_dataset_cosine_losses)
    same_dataset_cosine_loss = sum(same_dataset_cosine_losses) / len(same_dataset_cosine_losses)
    return (0.2 * between_dataset_cosine_loss) + (0.8 * same_dataset_cosine_loss)



def train(
    model: torch.nn.Module,
    data_loaders: List[DataLoader],
    optimizer: torch.optim,
    reconstruction_loss: torch.nn,
    args: Dict,
    device: torch.device,
):
    model.train()
    # Training loop
    for epoch in range(args.num_epochs):
        for batches in zip(*data_loaders):  # Load one batch from each dataset
            batches = [batch[0].to(device) for batch in batches]
            # Forward pass
            latents = []
            recons = []
            for batch in batches:
                latent, recon = model(batch)
                latents.append(latent)
                recons.append(recon)

            # Compute losses
            loss_recon = sum(
                reconstruction_loss(recon, batch)
                for recon, batch in zip(recons, batches)
            )

            # Compute cosine similarity loss
            loss_cosine = multi_dataset_cosine_similarity_loss(latents)

            loss = (
                loss_recon + (0.1 * loss_cosine)
            )  # Adjust weight for cosine similarity loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Logging
        if args.use_wandb:
                wandb.log(data={"train_loss": loss.item()}, step=epoch)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {loss.item():.8f}")


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_hyperparameters()
    if args.use_wandb:
        print(f"{args.use_wandb=}")
        wandb.init(project=args.project,
                    name=args.run,
                    config=dict(args))
    data_loaders, dataset_names, root = get_dataloaders(args)
    # Model, optimizer, and loss function
    model = Autoencoder(args.input_dim, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    reconstruction_loss = nn.MSELoss().to(device)
    train(model, data_loaders, optimizer, reconstruction_loss, args, device)
    # After training
    torch.save(model.state_dict(), f"{root}/autoencoder/autoencoder_state_dict.pth")


if __name__ == "__main__":
    main()
