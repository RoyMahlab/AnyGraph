import os
import torch
import random
import numpy as np
import scipy


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
    print("set seed")


seeder()
matrix_size = 48362
nnz = 781_000


# Set device to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

indices = torch.randint(0, matrix_size, (2, nnz), dtype=torch.long).to(device)
values = torch.randn(nnz).to(device)
sparse_matrix = torch.sparse_coo_tensor(
    indices, values, (matrix_size, matrix_size), device=device, dtype=torch.float64
).coalesce()

print(f"sparse_matrix = {sparse_matrix}")
q = 16
niter = 100
coo_matrix = scipy.sparse.coo_matrix(
    (
        sparse_matrix.values().numpy(),
        (sparse_matrix.indices()[0].numpy(), sparse_matrix.indices()[1].numpy()),
    ),
    shape=sparse_matrix.shape,
)
# do svds
# U1, S1, V1 = torch.svd_lowrank(sparse_matrix.to_dense(), q=q, niter=niter)
# U1, S1, V1 = torch.linalg.svd(sparse_matrix.to_dense(), full_matrices=False)
U1, S1, V1 = scipy.sparse.linalg.svds(coo_matrix, k=q, maxiter=niter)
seeder()
U2, S2, V2 = scipy.sparse.linalg.svds(coo_matrix, k=q, maxiter=niter)
# U2, S2, V2 = torch.linalg.svd(sparse_matrix.to_dense(), full_matrices=False)
# U2, S2, V2 = torch.svd_lowrank(sparse_matrix.to_dense(), q=q, niter=niter)

# Print results
print(f"U1.sum : {U1.sum().item()}")
print(f"U2.sum : {U2.sum().item()}")
print(f"U1.sum == U2.sum : {(U2.sum() == U1.sum()).item()}")
