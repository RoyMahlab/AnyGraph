import torch
from data_loader import get_data
from typing import List
from pathlib import Path
from autoencoder import Autoencoder

def load_model(root: str) -> torch.nn.Module:
    return torch.load(f"{root}/autoencoder/autoencoder.pth")

def get_latent_representations(model: Autoencoder, dataset: torch.Tensor, device: torch.device) -> torch.Tensor:
    latent, pred = model(dataset.to(device))
    return latent

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, dataset_names, root = get_data()
    autoencoder = load_model(root)
    autoencoder.eval()
    output_dir = "features_latent_representations"
    file_name = "latent_representations.pth"
    new_data_path = Path(root + f'/{output_dir}')
    for dataset, dataset_name in zip(datasets, dataset_names):
        print(f"Processing {dataset_name}")
        with torch.no_grad():
            latent_representations = get_latent_representations(autoencoder, dataset, device)
        dataset_output_dir = new_data_path / Path(f"{dataset_name}/{file_name}")
        dataset_output_dir.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latent_representations, dataset_output_dir)
    
if __name__ == "__main__":
    main()