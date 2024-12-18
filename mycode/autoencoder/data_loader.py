import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import rootutils
from typing import List, Tuple, Dict


def get_root_directory() -> str:
    # Automatically find and set the root directory
    root = rootutils.setup_root(
        search_from=__file__,  # Start searching from the current file location
    ).__str__()
    print(f"root = {root}")
    return root


def get_data(data_dir: str) -> List[torch.Tensor]:
    # Load data
    datasets, dataset_names = [], []
    root = get_root_directory()
    dir_path = Path(root) / Path(data_dir) #"/feat_matrices_svd")
    for dir in dir_path.iterdir():
        for file in dir.iterdir():
            matrix = torch.load(file)
            dataset_names.append(dir.name)
            datasets.append(matrix)
    return datasets, dataset_names, root


def get_dataloaders(args: Dict) -> Tuple[List[DataLoader], List[str], str]:
    datasets, dataset_names, root = get_data(args.data_dir)
    data_loaders = []
    for dataset in datasets:
        dataset = TensorDataset(dataset)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        data_loaders.append(data_loader)
    return data_loaders, dataset_names, root
