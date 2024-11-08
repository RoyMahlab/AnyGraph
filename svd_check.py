import torch

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Matrix dimensions
rows, cols = 10000, 5000

# Sparsity settings
density = 0.001  # Adjust the density as needed for sparsity

# Number of non-zero elements based on density
num_nonzeros = int(density * rows * cols)

# Random indices for non-zero elements in sparse COO format
indices = torch.randint(0, rows, (2, num_nonzeros), device=device)

# Random values for these indices
values = torch.randn(num_nonzeros, device=device)

# Create sparse COO tensor and move it to the GPU
sparse_matrix = torch.sparse_coo_tensor(indices, values, (rows, cols), device=device)


# Parameters for low-rank SVD
q = 512    # Rank for approximation
niter = 2  # Number of power iterations for accuracy

# Perform low-rank SVD on the dense matrix
U, S, V = torch.svd_lowrank(sparse_matrix, q=q, niter=niter)

# Reconstruct the approximate matrix from U, S, and V on the GPU
approx_matrix = U @ torch.diag(S) @ V.T

# Print shapes of results
print("Original dense matrix shape:", sparse_matrix.sum())
print("U shape:", U.sum())  # U should be (10000, 512)
print("S shape:", S.sum())  # S should be (512,)
print("V shape:", V.sum())  # V should be (5000, 512)
print("Approximate matrix shape:", approx_matrix.sum())
