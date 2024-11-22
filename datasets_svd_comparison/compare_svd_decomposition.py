import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
from my_utils import get_root_directory
root = get_root_directory()

def load_data() -> Tuple[List, List]:
    data = []
    dataset_names = []
    dir_path = Path(root + f'/adj_matrices_svd')
    for dir in dir_path.iterdir():
        for file in dir.iterdir():
            matrix = torch.load(file).numpy()
            dataset_names.append(dir.name)
            data.append(matrix)
    return data, dataset_names
            
def flatten_matrices(data: list) -> list:
    return list(map(lambda matrix: matrix.flatten(), data))

def plot_histograms(data: list, dataset_names: list):
    bins = 100
    alpha = 0.5
    for j in range(2):
        plt.figure(figsize=(10, 6))
        for matrix, dataset_name in zip(data[j*5:j*5+5], dataset_names[j*5:j*5+5]):
            mean = np.mean(matrix)
            variance = np.var(matrix)
            plt.hist(matrix, bins=bins, alpha=alpha,
                     range=(-0.02, 0.02),
                     label=f'{dataset_name} - mean {mean:.4f}, var {variance:.8f}',
                     edgecolor='black')
        plt.title('Overlaid Histogram of Matrices')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"histograms_{j}.png")
    
def main():
    data, dataset_names = load_data()
    data = flatten_matrices(data)
    plot_histograms(data, dataset_names)

if __name__ == '__main__':
    main()