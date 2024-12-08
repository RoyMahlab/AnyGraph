#!/bin/bash

for i in 512 256 128 64 32  # Replace 1..10 with your desired range
do
    echo "Generating svd vectors for latent dimension ${i}"
    python main.py \
    --dataset link2 --latdim ${i} \
    --save_feature_matrices_path feat_matrices_svd_${i}

    if [ $? -ne 0 ]; then
        echo "Error: main feat_matrices_svd_${i} failed. Exiting."
        exit 1
    fi
    echo ""

    echo "Trainig autoencoder for latent dimension ${i}"
    python autoencoder/autoencoder_aligner.py \
    --data_path feat_matrices_svd_${i} --latent_size ${i}

    if [ $? -ne 0 ]; then
        echo "Error: autoencoder/autoencoder_aligner.py feat_matrices_svd_512_${i} failed. Exiting."
        exit 1
    fi
    echo ""

    echo "Creating data from trained autoencoder for latent dimension ${i}"
    python autoencoder/create_new_data.py \
    --data_path feat_matrices_svd_${i} --latent_size ${i}
    if [ $? -ne 0 ]; then
        echo "Error: autoencoder/create_new_data.py ${i} failed. Exiting."
        exit 1
    fi
    echo ""

    echo "Training Photo for latent dimension ${i}"
    python main.py \
    --save_path Photo_latent_dim_${i} \
    --dataset Photo --latdim ${i} \
    --latent_features_path features_latent_representations_${i} \
    --project anygraph_Photo --run latent_space_${i}_from_scratch \
    --use_wandb

    if [ $? -ne 0 ]; then
        echo "Error: main.py ${i} failed. Exiting."
        exit 1
    fi
    echo ""

    echo "Training cora for latent dimension ${i}"
    python main.py \
    --save_path cora_latent_dim_${i} \
    --dataset cora --latdim ${i} \
    --latent_features_path features_latent_representations_${i} \
    --project anygraph_cora --run latent_space_${i}_from_scratch \
    --use_wandb

    if [ $? -ne 0 ]; then
        echo "Error: main.py ${i} failed. Exiting."
        exit 1
    fi
    echo ""

    echo "Training cora for latent dimension ${i}"
    python main.py \
    --load_model Photo_latent_dim_${i} \
    --save_path cora_latent_dim_${i}_Photo_pretrained \
    --dataset cora --latdim ${i} \
    --latent_features_path features_latent_representations_${i} \
    --project anygraph_cora --run latent_space_${i}_from_Photo \
    --use_wandb

    if [ $? -ne 0 ]; then
        echo "Error: main.py ${i} failed. Exiting."
        exit 1
    fi
    echo ""

    echo "Training arxiv for latent dimension ${i}"
    python main.py \
    --save_path arxiv_latent_dim_${i} \
    --dataset arxiv --latdim ${i} \
    --latent_features_path features_latent_representations_${i} \
    --project anygraph_arxiv --run latent_space_${i}_from_scratch \
    --use_wandb

    if [ $? -ne 0 ]; then
        echo "Error: main.py ${i} failed. Exiting."
        exit 1
    fi
    echo ""

    echo "Training arxiv for latent dimension ${i}"
    python main.py \
    --load_model Photo_latent_dim_${i} \
    --save_path arxiv_latent_dim_${i}_Photo_pretrained \
    --dataset arxiv --latdim ${i} \
    --latent_features_path features_latent_representations_${i} \
    --project anygraph_arxiv --run latent_space_${i}_from_Photo \
    --use_wandb

    if [ $? -ne 0 ]; then
        echo "Error: main.py ${i} failed. Exiting."
        exit 1
    fi

done