import os
import torch
import random
import numpy as np

def seeder():
    # Ensure deterministic behavior for cuBLAS
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Set random seeds for reproducibility
    # Set seeds again before each call
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: Force single-threaded execution
    torch.set_num_threads(1)
    
seeder()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

indices = torch.randint(0, 48362, (2, 781_000), dtype=torch.long).to(device)  # 10000 non-zero elements
values = torch.randn(781_000).to(device)  # Random values for the non-zero elements
sparse_matrix = torch.sparse_coo_tensor(indices, values, (48362, 48362), device=device)

print(f"sparse_matrix = {sparse_matrix}")
# Parameters for low-rank SVD
q = 512    # Rank for approximation

# Try disabling power iterations
niter = 0

# Perform low-rank SVD on the dense matrix
U1, S1, V1 = torch.svd_lowrank(sparse_matrix, q=q, niter=niter)
seeder()
U2, S2, V2 = torch.svd_lowrank(sparse_matrix, q=q, niter=niter)

# Print results
print(f"{U1.sum()=}")
print(f"{U2.sum()=}")
print(f"{U2.sum() == U1.sum()=}")
