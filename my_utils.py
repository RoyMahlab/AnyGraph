import rootutils
import wandb
import torch
import numpy as np
import random

def get_root_directory() -> str:
    # Automatically find and set the root directory
    root = rootutils.setup_root(
        search_from=__file__,  # Start searching from the current file location
    ).__str__()
    print(f"root = {root}")
    return root

def initialize_wandb(args:dict) -> None:
    wandb.init(project=args.project,
               name=args.run,
               config=args)
    
def initialize_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Initialized seed to {seed}")
    
def convert_torch_tensor_to_torch_sparse_coo_tensor(tensor: torch.Tensor) -> torch.sparse_coo_tensor:
    indices = tensor.nonzero().t()
    values = tensor[tensor.nonzero().t().unbind()]
    return torch.sparse_coo_tensor(indices, values, tensor.size())