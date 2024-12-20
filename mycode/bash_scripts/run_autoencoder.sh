#!/bin/bash

check_error() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed. Exiting."
        exit 1
    else 
        echo "$1 executed successfully."
    fi

}

latent_size=32

# Run autoencoder_aligner.py
python mycode/autoencoder/autoencoder_aligner.py --data_path mycode/data/feat_matrices_svd_$latent_size \
 --model_filename autoencoder_gnn_state_dict_${latent_size} \
 --latent_size $latent_size --use_edge_index

check_error autoencoder_aligner.py

# Run create_new_data.py
python mycode/autoencoder/create_new_data.py --data_path mycode/data/feat_matrices_svd_$latent_size \
 --model_filename autoencoder_gnn_state_dict_${latent_size} \
 --latent_size $latent_size \
 --output_dir mycode/data/features_latent_representations_gnn_${latent_size}\
 --use_edge_index

check_error create_new_data.py

# Run compare_features_decomposition.py
python mycode/svd_comparison/compare_features_decomposition.py --data_folder \
 mycode/data/features_latent_representations_gnn_$latent_size \
 --output_folder mycode/svd_comparison/latent_feat_heat_maps_$latent_size

check_error compare_features_decomposition.py

echo "All scripts executed successfully."
