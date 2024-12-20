import torch
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from compare_svd_decomposition import load_data
from typing import List, Tuple, Union

def normalize_single_matrix(matrix: np.array):
    col_norms = np.linalg.norm(matrix, axis=0, keepdims=True) + 1e-8
    normalized_matrix = matrix / col_norms
    return normalized_matrix


def cosine_similarity(x: np.array, y: np.array) -> np.array:
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[0] == y.shape[0]
    return x.T @ y


def pad_matrices(matrix_i: np.array, matrix_j: np.array) -> np.array:
    # Pad the matrices with zeros to make them the same size
    H_i, W_i = matrix_i.shape
    H_j, W_j = matrix_j.shape
    if H_i < H_j:
        matrix_i = np.vstack((matrix_i, np.zeros((H_j - H_i, W_i))))
    elif H_i > H_j:
        matrix_j = np.vstack((matrix_j, np.zeros((H_i - H_j, W_j))))
    return matrix_i, matrix_j


def plot_heatmaps(
    data: List[np.array],
    dataset_names: List[str],
    output_dir: str,
    data_type: str,
) -> None:
    # Plot the heatmap
    couples = set()
    for matrix_i, dataset_name_i in zip(data, dataset_names):
        matrix_i = normalize_single_matrix(matrix_i)
        for matrix_j, dataset_name_j in zip(data, dataset_names):
            if tuple(sorted(([dataset_name_i, dataset_name_j]))) in couples:
                continue
            couples.add(tuple(sorted([dataset_name_i, dataset_name_j])))
            plt.figure(figsize=(20, 18))
            print("Processing ", dataset_name_j)
            matrix_j = normalize_single_matrix(matrix_j)
            matrix_i, matrix_j = pad_matrices(matrix_i, matrix_j)
            matrix = cosine_similarity(matrix_i, matrix_j)
            sns.heatmap(
                matrix,
                vmin=-1,
                vmax=1,
                annot=True,
                fmt=".5f",
                cmap="coolwarm",
                cbar=True,
                cbar_kws={"shrink": 0.8},
            )
            plt.title(
                f"Heatmap of {dataset_name_i} vs {dataset_name_j} {data_type} "
                f"matrix singular vectors cosine similarity"
            )
            plt.xlabel("Singular vector index")
            plt.ylabel("Singular vector index")
            plt.savefig(f"{output_dir}/heatmap_{dataset_name_i}_{dataset_name_j}.png")
            plt.close()


def create_folder(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Model Parameters")
    parser.add_argument(
        "--data_folder", default=".", type=str, help="path to data"
    )
    parser.add_argument(
        "--output_folder", default=".", type=str, help="path to put images in"
    )
    parser.add_argument(
        "--data_type", default="features", type=str, help="what to put on the plots"
    )
    args = parser.parse_args()
    create_folder(args.output_folder)
    data, dataset_names, root = load_data(args.data_folder)
    output_path = f"{root}/{args.output_folder}"
    plot_heatmaps(data, dataset_names, output_path, args.data_type)


if __name__ == "__main__":
    main()
