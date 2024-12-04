import torch
from data_loader import get_data
from typing import List
from pathlib import Path
from autoencoder import Autoencoder
from hyperparameters import get_hyperparameters
from omegaconf import OmegaConf


def load_model(root: str, args: OmegaConf, device: torch.device) -> torch.nn.Module:
    model = Autoencoder(args.input_dim, args.latent_dim).to(device)
    model.load_state_dict(torch.load(f"{root}/autoencoder/autoencoder_state_dict.pth"))
    return model


def get_latent_representations(
    model: Autoencoder, dataset: torch.Tensor, device: torch.device
) -> torch.Tensor:
    latent, pred = model(dataset.to(device))
    return latent


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_hyperparameters()
    datasets, dataset_names, root = get_data(args.data_dir)
    autoencoder = load_model(root, args, device)
    autoencoder.eval()
    output_dir = "features_latent_representations_512"
    file_name = "Photo_latent_representations_512.pth"
    new_data_path = Path(root + f"/{output_dir}")
    for dataset, dataset_name in zip(datasets, dataset_names):
        if dataset_name != "Photo":
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
    main()
