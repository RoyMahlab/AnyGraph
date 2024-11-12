import torch
import random
import numpy as np

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.use_deterministic_algorithms(True)

# Create a sparse tensor
indices = torch.tensor([[0, 1], [1, 0]])
values = torch.tensor([2.0, 3.0])
sparse_tensor = torch.sparse_coo_tensor(indices, values, (2, 2))

other_tensor = torch.randn(2, 2)
# Check reproducibility across runs
results = []
for _ in range(5):
    result = torch.sparse.mm(sparse_tensor, other_tensor)
    results.append(result)

# Validate consistency
consistent = all(torch.equal(results[0], r) for r in results[1:])
print(f"Reproducibility test passed: {consistent}")
