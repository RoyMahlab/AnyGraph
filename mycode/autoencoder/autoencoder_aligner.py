import random
import numpy as np
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as PygDataLoader
from autoencoder import Autoencoder, AutoencoderGNN
from data_loader import get_dataloaders, get_graph_dataloaders
import argparse
from hyperparameters import get_hyperparameters
from typing import Union, List, Dict, Tuple


def pad_matrices(
    matrix_i: torch.tensor, matrix_j: torch.tensor
) -> Tuple[torch.tensor, torch.tensor]:
    # Pad the matrices with zeros to make them the same size
    H_i, W_i = matrix_i.shape
    H_j, W_j = matrix_j.shape
    if H_i < H_j:
        matrix_i = torch.vstack(
            (matrix_i, torch.zeros((H_j - H_i, W_i), device=matrix_i.device))
        )
    elif H_i > H_j:
        matrix_j = torch.vstack(
            (matrix_j, torch.zeros((H_i - H_j, W_j), device=matrix_i.device))
        )
    return matrix_i, matrix_j


def multi_dataset_cosine_similarity_loss(
    latents: List, use_edge_index: bool
) -> torch.tensor:
    # normalize latents
    latents = [latent / latent.norm(dim=0, keepdim=True) + 1e-8 for latent in latents]

    same_dataset_cosine_losses = []
    between_dataset_cosine_losses = []
    for i in range(len(latents)):
        latent_i = latents[i]
        cosine_matrix = torch.mm(latent_i.t(), latent_i)
        diagonal = torch.eye(cosine_matrix.shape[0], device=latent_i.device)

        # Penalize similarity between datasets
        same_dataset_cosine_losses.append((cosine_matrix - diagonal).mean())
        for j in range(i + 1, len(latents)):
            latent_j = latents[j]
            if use_edge_index:
                latent_i, latent_j = pad_matrices(latent_i, latent_j)
            cosine_matrix = torch.mm(latent_i.t(), latent_j)

            # Penalize disimilarity between datasets
            between_dataset_cosine_losses.append(1 - cosine_matrix.mean())

    between_dataset_cosine_loss = sum(between_dataset_cosine_losses) / len(
        between_dataset_cosine_losses
    )
    same_dataset_cosine_loss = sum(same_dataset_cosine_losses) / len(
        same_dataset_cosine_losses
    )
    return (between_dataset_cosine_loss) + (same_dataset_cosine_loss)


def compute_reconstruction_loss(recons, batches, reconstruction_loss, use_edge_index):
    """
    Computes the total reconstruction loss for a set of reconstructions and batches.

    Parameters:
    - recons (list): List of reconstructed outputs.
    - batches (list): List of input batches corresponding to the reconstructions.
    - reconstruction_loss (function): A function to calculate the reconstruction loss
      between a single reconstructed output and a batch.
    - use_edge_index (bool): Flag to determine if edge_index should be used in loss calculation.

    Returns:
    - float: Total reconstruction loss.
    """
    loss_recon = sum(
        (
            reconstruction_loss(recon, batch.x)
            if use_edge_index
            else reconstruction_loss(recon, batch)
        )
        for recon, batch in zip(recons, batches)
    )
    return loss_recon


def train(
    model: torch.nn.Module,
    data_loaders: List[Union[PygDataLoader, DataLoader]],
    optimizer: torch.optim,
    reconstruction_loss: torch.nn,
    args: Dict,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
):
    loss = 0
    model.train()

    # Early stopping parameters
    patience = 5  # Number of epochs to wait for loss improvement
    min_delta = 1e-4  # Minimum improvement to qualify as a decrease
    best_loss = float("inf")
    epochs_no_improve = 0

    # Training loop
    for epoch in range(args.num_epochs):
        for i, batches in enumerate(
            zip(*data_loaders)
        ):  # Load one batch from each dataset
            batches = [batch[0].to(device) for batch in batches]
            # Forward pass
            latents = []
            recons = []

            for batch in batches:
                if args.use_edge_index:
                    latent, recon = model(batch.x, batch.edge_index)
                else:
                    latent, recon = model(batch)
                latents.append(latent)
                recons.append(recon)

            loss_recon = compute_reconstruction_loss(
                recons, batches, reconstruction_loss, args.use_edge_index
            )

            # Compute cosine similarity loss
            loss_cosine = multi_dataset_cosine_similarity_loss(
                latents, args.use_edge_index
            )

            loss = loss_recon + (
                2 * loss_cosine
            )  # Adjust weight for cosine similarity loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i == 100:
                break
        # Logging
        if args.use_wandb:
            wandb.log(data={"train_loss": loss.item()}, step=epoch)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {loss.item():.8f}")
        scheduler.step()
        # Check early stopping condition
        if best_loss - loss.item() > min_delta:
            best_loss = loss.item()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break


def main(general_args: Dict):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_hyperparameters()
    args.data_dir = general_args.data_path
    args.input_dim = general_args.latent_size
    args.latent_dim = general_args.latent_size
    args.use_edge_index = general_args.use_edge_index
    if args.use_wandb:
        print(f"{args.use_wandb=}")
        wandb.init(project=args.project, name=args.run, config=dict(args))
    if args.use_edge_index:
        data_loaders, dataset_names, root = get_graph_dataloaders(args)
    else:
        data_loaders, dataset_names, root = get_dataloaders(args)
    # Model, optimizer, and loss function
    print(f"args.input_dim: {args.input_dim}, args.latent_dim: {args.latent_dim}")
    model = (
        AutoencoderGNN(args.input_dim, args.latent_dim).to(device)
        if args.use_edge_index
        else Autoencoder(args.input_dim, args.latent_dim).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    reconstruction_loss = nn.MSELoss().to(device)
    scheduler = StepLR(optimizer, step_size=args.num_epochs // 50, gamma=args.gamma)
    train(model, data_loaders, optimizer, reconstruction_loss, args, device, scheduler)
    # After training
    torch.save(
        model.state_dict(),
        f"{root}/mycode/autoencoder/{general_args.model_filename}.pth",
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    parser = argparse.ArgumentParser(description="Model Parameters")
    parser.add_argument("--data_path", default=".", type=str, help="path to data")
    parser.add_argument("--model_filename", default=".", type=str, help="path to data")
    parser.add_argument(
        "--latent_size", default=512, type=int, help="latent dimensionality"
    )
    parser.add_argument(
        "--use_edge_index",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="use edge index",
    )
    general_args = parser.parse_args()
    main(general_args)
