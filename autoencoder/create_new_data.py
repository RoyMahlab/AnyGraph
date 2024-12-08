import torch
from data_loader import get_data
from typing import List, Dict
from pathlib import Path
from autoencoder import Autoencoder
from hyperparameters import get_hyperparameters
from omegaconf import OmegaConf
import argparse


def load_model(
    root: str, args: OmegaConf, device: torch.device, latent_size: int
) -> torch.nn.Module:
    model = Autoencoder(args.input_dim, args.latent_dim).to(device)
    model.load_state_dict(
        torch.load(f"{root}/autoencoder/autoencoder_state_dict_{latent_size}.pth")
    )
    return model


def get_latent_representations(
    model: Autoencoder, dataset: torch.Tensor, device: torch.device
) -> torch.Tensor:
    latent, pred = model(dataset.to(device))
    return latent


def main(general_args: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_hyperparameters()
    args.input_dim = general_args.latent_size
    args.latent_dim = general_args.latent_size
    datasets, dataset_names, root = get_data(general_args.data_path)
    autoencoder = load_model(root, args, device, general_args.latent_size)
    autoencoder.eval()
    output_dir = f"features_latent_representations_{general_args.latent_size}"
    file_name = "latent_representations.pth"
    new_data_path = Path(root + f"/{output_dir}")
    for dataset, dataset_name in zip(datasets, dataset_names):
        if (
            dataset_name != "Photo"
            and dataset_name != "cora"
            and dataset_name != "arxiv"
        ):
            continue
        print(f"Processing {dataset_name}")
        with torch.no_grad():
            latent_representations = get_latent_representations(
                autoencoder, dataset, device
            )
        dataset_output_dir = new_data_path / Path(f"{dataset_name}/{file_name}")
        dataset_output_dir.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latent_representations, dataset_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Parameters")
    parser.add_argument(
        "--data_path", default="features_matrices_svd", type=str, help="path to data"
    )
    parser.add_argument(
        "--latent_size", default=512, type=int, help="latent dimensionality"
    )
    general_args = parser.parse_args()
    main(general_args)
