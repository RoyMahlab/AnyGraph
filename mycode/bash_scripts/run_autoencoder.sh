#!/bin/bash

# Run autoencoder_aligner.py
python autoencoder/autoencoder_aligner.py
if [ $? -ne 0 ]; then
    echo "Error: autoencoder_aligner.py failed. Exiting."
    exit 1
fi

# Run create_new_data.py
python autoencoder/create_new_data.py
if [ $? -ne 0 ]; then
    echo "Error: create_new_data.py failed. Exiting."
    exit 1
fi

# Run compare_features_decomposition.py
# python datasets_svd_comparison/compare_features_decomposition.py
# if [ $? -ne 0 ]; then
#     echo "Error: compare_features_decomposition.py failed. Exiting."
#     exit 1
# fi

echo "All scripts executed successfully."
