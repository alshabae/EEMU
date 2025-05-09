# EEMU

Source for "Meta-Learn to Unlearn: Enhanced Exact Machine Unlearning in Recommendation Systems with Meta-Learning"

## Create the conda environment

conda create --name ENV_NAME --file conda_requirements.txt python=3.7 \
conda activate ENV_NAME \
conda install pip \
python3 -m pip install -r requirements.txt

## Downloading the datasets

We use public benchmark datasets including: [MovieLens1m](https://grouplens.org/datasets/movielens/1m/). Download the datasets into a datasets subdirectory as follows: PATH/TO/DATASETS_DIR/ml-1m, and subsequently seting the following environment variable:

export DATASETS_DIR=PATH/TO/DATASETS

## Reproducing Results

The main results in the paper for for EEMU can be reproduced by running the following command:

TOKENIZERS_PARALLELISM=false torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=8 main.py --dataset=ml1m --device=gpu --batch_size=32 --print_freq=16 --lr=2e-4 --epochs=100 --margin=1 --warm_threshold=0.2 --num_workers=6 --feat_embed_dim=96 --user_embed_hidden_dims=512,256,128,64 --item_embed_hidden_dims=256,128,64 --usable_user_feats=id,occ,age,gender --k=10 --num_shards=8 --query_set_length=1 --heuristic_sample_size=7 --heuristic=all_biased_sample --dataset_dir=DATASETS_DIR

## Hardware Requiremnts

The code is designed to run on multiple GPUs, with one GPU required per shard in your experiment. The training time for all shards in a single epoch depends on the dataset, but typically takes a few minutes on a desktop-class GPU. The code has been tested on a machine with over 100GB of RAM, which allows data loading to be efficient.
