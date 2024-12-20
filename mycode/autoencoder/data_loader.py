import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data, Batch
from subgraph_dataloader import get_subgraph_dataloader
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
    dir_path = Path(root) / Path(data_dir)  # "/feat_matrices_svd")
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
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        data_loaders.append(data_loader)
    return data_loaders, dataset_names, root


def convert_sparse_adj_matrix_to_edge_index(adj_matrix: torch.Tensor) -> torch.Tensor:
    row, col = adj_matrix.nonzero()  # Get the nonzero indices
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    return edge_index


def load_edge_index(dataset_name: str, root: str) -> torch.Tensor:
    # Load adjacency matrices
    with open(
        Path(root) / Path("datasets") / Path(dataset_name) / Path("trn_mat.pkl"), "rb"
    ) as f:
        adj_matrix = pickle.load(f)
    edge_index = convert_sparse_adj_matrix_to_edge_index(adj_matrix)
    return edge_index


def get_graph_dataloaders(args: Dict) -> Tuple[List[DataLoader], List[str], str]:
    datasets, dataset_names, root = get_data(args.data_dir)
    data_loaders = []
    for i, dataset in enumerate(datasets):
        edge_index = load_edge_index(dataset_names[i], root)
        data = Data(x=dataset, edge_index=edge_index)
        subgraph_data_loader = get_subgraph_dataloader(data, 1_000, 1, True)
        data_loaders.append(subgraph_data_loader)
    return data_loaders, dataset_names, root


def get_dataloaders(args: Dict) -> Tuple[List[DataLoader], List[str], str]:
    datasets, dataset_names, root = get_data(args.data_dir)
    data_loaders = []
    for dataset in datasets:
        dataset = TensorDataset(dataset)
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        data_loaders.append(data_loader)
    return data_loaders, dataset_names, root
