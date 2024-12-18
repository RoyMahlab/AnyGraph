from omegaconf import OmegaConf


def get_hyperparameters():
    return OmegaConf.create(
        {
            "input_dim": 512,
            "latent_dim": 512,
            "learning_rate": 1e-3,
            "num_epochs": 100,
            "batch_size": 64,
            "use_wandb": False,
            "project": "autoencoder",
            "run": "deeper_autoencoder",
            "data_dir": "feat_matrices_svd_512"
        }
    )
