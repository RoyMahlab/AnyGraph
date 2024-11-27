import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
import os
import sys

# Add the path to 'external_modules' to sys.path
parent_dir = os.path.abspath(".")
print(parent_dir)
sys.path.append(parent_dir)
from my_utils import get_root_directory

root = get_root_directory()


def load_data(dir: str) -> Tuple[List[np.array], List[str], str]:
    data = []
    dataset_names = []
    dir_path = Path(root + f"/{dir}")
    for dir in dir_path.iterdir():
        for file in dir.iterdir():
            matrix = torch.load(file).numpy()
            dataset_names.append(dir.name)
            data.append(matrix)
    return data, dataset_names, root


def flatten_matrices(data: list) -> list:
    return list(map(lambda matrix: matrix.flatten(), data))


def plot_histograms(
    data: list, dataset_names: list, jump_size: int, data_type: str, output_path: str
) -> None:
    bins = 100
    alpha = 0.5
    for j in range(2):
        plt.figure(figsize=(10, 6))
        for matrix, dataset_name in zip(
            data[j * jump_size :], dataset_names[j * jump_size :]
        ):
            mean = np.mean(matrix)
            variance = np.var(matrix)
            plt.hist(
                matrix,
                bins=bins,
                alpha=alpha,
                range=(-0.02, 0.02),
                label=f"{dataset_name} - mean {mean:.4f}, var {variance:.8f}",
                edgecolor="black",
            )
        plt.title(f"{data_type} matrix singular values combined histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_path}/histograms_{j}.png")
        plt.close()


def get_data(data_type: str) -> Tuple[str, str, int]:
    if data_type == "features":
        data_folder = "feat_matrices_svd"
        output_folder = "datasets_svd_comparison/feat_histograms"
        jump_size = 3
    elif data_type == "adjacency":
        data_folder = "adj_matrices_svd_16"
        output_folder = "datasets_svd_comparison/adj_histograms"
        jump_size = 5
    else:
        raise ValueError("Invalid data type")
    return data_folder, output_folder, jump_size


def create_folder(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    data_type = "features"
    data_folder, output_folder, jump_size = get_data(data_type)
    data, dataset_names, root = load_data(data_folder)
    output_path = f"{root}/{output_folder}"
    create_folder(output_path)
    data = flatten_matrices(data)
    plot_histograms(data, dataset_names, jump_size, data_type, output_path)


if __name__ == "__main__":
    main()
