import rootutils
import wandb

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