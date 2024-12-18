python main.py \
 --save_path arxiv_latent_512_from_scratch \
 --dataset link2 --latdim 512 \
 --latent_features_path features_latent_representations_512 \
 --project anygraph_arxiv --run latent_space_512_from_scratch \
 --use_wandb

if [ $? -ne 0 ]; then
    echo "Error: latent_space_512_from_scratch failed. Exiting."
    exit 1
fi

python main.py \
 --load_model Photo_trained_using_autoencoder \
 --save_path arxiv_latent_512_from_photo_autoencoder \
 --dataset link2 --latdim 512 \
 --latent_features_path features_latent_representations_512 \
 --project anygraph_arxiv --run latent_space_512_from_photo_autoencoder \
 --use_wandb

if [ $? -ne 0 ]; then
    echo "Error: latent_space_512_from_photo_autoencoder failed. Exiting."
    exit 1
fi
