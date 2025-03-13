import copy
import pickle
import sys
import os
import setproctitle

from utils import batch_to_device, get_dataloaders, read_data


sys.path.append(os.path.join(sys.path[0], "cpp_extensions", "build"))

from dist_eval import inference
from model.sparsenn import SparseNN

from absl import app, flags
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.backends.cudnn
import wandb
from copy import deepcopy
import torch.distributed as dist


FLAGS = flags.FLAGS
# General Config
flags.DEFINE_integer("seed", 0, "Random seed for all modules")
flags.DEFINE_integer("print_freq", 32, "How often to log losses")
flags.DEFINE_integer("test_freq", 1, "How frequently to test")
flags.DEFINE_string("device", "cpu", "Specify whether to use the CPU or GPU")
flags.DEFINE_bool("wandb_logging", False, "Whether or not to log metrics to wandb")
flags.DEFINE_string("wandb_project", "ColdStartRecommendation", "Name of the project in wandb")
flags.DEFINE_string("run_name", "Run0", "Name of the run to log in wandb")
flags.DEFINE_integer("k", 10, "TopK value")

# Data Config
flags.DEFINE_string("dataset_dir", '/home/keshav/datasets', "directory to store and load datasets from")
flags.DEFINE_string("dataset", "ml1m", "Dataset to use")
flags.DEFINE_integer("user_core", 20, "User Core Setting")
flags.DEFINE_bool("load_edge_lccsr", False, "Whether to load compact adjacency matrix")
flags.DEFINE_string("edge_lccsr_path", None, "Path to load compact adj from disk")
flags.DEFINE_integer("num_workers", 8, "Number of dataloader processes")
flags.DEFINE_float("warm_threshold", 0.2, "Fraction of warm items")
flags.DEFINE_bool("order", False, "Whether to order samples by timestamp")
flags.DEFINE_bool("sampling_bias", False, "Whether or not to sample neighbors with inverse degree bias")
flags.DEFINE_float("sampling_softmax_temp", 1.0, "Sampling softmax temp")

# Model Config
flags.DEFINE_string("synthesis_model", None, "The type of set embedding model to use")
flags.DEFINE_string("recurr_mode", "gru", "Whether to user a GRU or LSTM for recurrent module")
flags.DEFINE_list("usable_user_feats", [], "Features that are usable")
flags.DEFINE_bool("user_feat_attn", False, "Whether to use feat attention between user features to predict unusable features")

# Model Hyperparameters
flags.DEFINE_integer("projection_dim", 64, "Multihead attention projection dim")
flags.DEFINE_integer("num_heads", 1, "Number of heads to be used in Multihead attention")
flags.DEFINE_integer("recurr_hidden_size", 32, "Recurrent module hidden size")
flags.DEFINE_integer("recurr_num_layers", 2, "GRU num layers")
flags.DEFINE_integer("feat_embed_dim", 96, "Embedding table dimensionality")
flags.DEFINE_integer("dense_feat_embed_dim", 384, "Embedding table dimensionality")
flags.DEFINE_list("user_embed_hidden_dims", [512, 256, 128], "Synthesis mode hidden dims")
flags.DEFINE_list("item_embed_hidden_dims", [256, 128], "Item tower MLP dims")
flags.DEFINE_integer("user_embed_dim", 128, "Synthesis Model output dim")

# Non-model Hyperparameters
flags.DEFINE_integer("batch_size", 64, "Batch Size")
flags.DEFINE_integer("item_inference_batch_size", 512, "Batch size to load items when computing all item representations before inference")
flags.DEFINE_float("lr", 1e-4, "Learning Rate")
flags.DEFINE_integer("epochs", 1, "Number of training epochs")
flags.DEFINE_float("margin", 1, "Margin used in the margin ranking loss")
flags.DEFINE_integer("num_negatives", 1, "Number of negative samples per positive")
flags.DEFINE_list("fanouts", [10], "comma separated list of fanouts")
flags.DEFINE_float("mfv_ratio", 0.0, "ratio of missing features in batch")
flags.DEFINE_integer("num_shards", 10, "number of training data shards")
flags.DEFINE_integer("query_set_length", 3, "number of edges in query set per user per shard")
flags.DEFINE_integer("heuristic_sample_size", 1, "number of edges in heuristic set per user per shard")
flags.DEFINE_string("heuristic", "other_shards_query", "Which heuristic to use for train_for_test set, options: other_shards_query, all, sample_only_cold, biased_sample")

def train(rank, device, user_feature_model, item_feature_model, num_user_feats, num_item_feats, user_feat_counts, item_feat_counts, usable_user_feats, unusable_user_feats, train_dataloader, train_for_test_dataloader, dataloader_val, inference_users_dataloader, inference_items_dataloader):
    model = SparseNN(
            user_feature_model,
            item_feature_model,
            num_user_feats=num_user_feats,
            num_item_feats=num_item_feats,
            user_feat_counts=user_feat_counts,
            item_feat_counts=item_feat_counts,
            usable_user_feats=usable_user_feats,
            unusable_user_feats=unusable_user_feats,
            user_feat_attn=FLAGS.user_feat_attn,
            feat_embed_dim=FLAGS.feat_embed_dim,
            dense_feat_embed_dim=FLAGS.dense_feat_embed_dim,
            user_embed_hidden_dims=FLAGS.user_embed_hidden_dims,
            item_embed_hidden_dims=FLAGS.item_embed_hidden_dims,
            user_embed_dim=FLAGS.user_embed_dim,
            seq_len=FLAGS.fanouts[0],
            recurr_hidden_size=FLAGS.recurr_hidden_size,
            recurr_num_layers=FLAGS.recurr_num_layers,
            recurr_mode=FLAGS.recurr_mode,
            projection_dim=FLAGS.projection_dim,
            nhead=FLAGS.num_heads,
            synthesis_model=FLAGS.synthesis_model,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    loss_fn = torch.nn.MarginRankingLoss(margin=FLAGS.margin)
    y = torch.tensor([1], device=device)
    
    num_samples = 0
    best_hr, best_ndcg, best_epoch= 0, 0, 0
    corr_cold_hr, corr_warm_hr, corr_cold_ndcg, corr_warm_ndcg = 0, 0, 0, 0

    for epoch in range(FLAGS.epochs):
        for i, (support_user_feats, query_user_feats, support_item_feats, query_item_feats) in enumerate(tqdm(train_dataloader, disable=True)):
            num_inner_iter = 1
            all_users_loss = 0

            weights_after = deepcopy(model.state_dict())
            weights_before = deepcopy(model.state_dict())
            for user in range(len(support_user_feats)):
                model.load_state_dict(weights_before)
                for _ in range(num_inner_iter):
                    user_feats_local, item_feats_local, neighbor_item_feats_local = batch_to_device(support_user_feats[user], support_item_feats[user], None, device)
                    spt_logits = model(user_feats_local, item_feats_local, neighbor_item_feats_local).flatten()
                    current_batch_size_local = int(user_feats_local['id'].shape[0] / ((2 * FLAGS.num_negatives) + 1))
                    spt_loss = loss_fn(spt_logits[:current_batch_size_local].repeat_interleave(2 * FLAGS.num_negatives), spt_logits[current_batch_size_local:], y)
                    optimizer.zero_grad()
                    spt_loss.backward()
                    optimizer.step()
                ###### end local update then start global update ############
                user_feats, item_feats, neighbor_item_feats = batch_to_device(query_user_feats[user], query_item_feats[user], None, device)
                qry_logits = model(user_feats, item_feats, neighbor_item_feats).flatten()
                current_batch_size = int(user_feats['id'].shape[0] / ((2 * FLAGS.num_negatives) + 1))
                qry_loss = loss_fn(qry_logits[:current_batch_size].repeat_interleave(2 * FLAGS.num_negatives), qry_logits[current_batch_size:], y)
                scaled_qry_loss = torch.div(qry_loss, FLAGS.batch_size)
                optimizer.zero_grad()
                scaled_qry_loss.backward()
                model.load_state_dict(weights_after)
                optimizer.step()
                weights_after = deepcopy(model.state_dict())
                all_users_loss = all_users_loss + scaled_qry_loss.clone().detach().cpu().item()

            num_samples += current_batch_size

            if FLAGS.wandb_logging and rank==0:
                wandb.log({
                    "Loss" : all_users_loss / num_samples
                })

            if (i + 1) % FLAGS.print_freq == 0:
                print(f"Epoch {epoch}, Iteration {i+1} / {len(train_dataloader)} - Average loss per user = {all_users_loss / num_samples}, rank {rank}")
                num_samples = 0

        # continue
        if (epoch + 1) % FLAGS.test_freq == 0:
            if rank != 0:
                inference(model, inference_items_dataloader, inference_users_dataloader, dataloader_val, train_for_test_dataloader, device, k=FLAGS.k, margin=FLAGS.margin, lr=FLAGS.lr, num_negatives=FLAGS.num_negatives, num_shards=FLAGS.num_shards, rank=rank)
                hr = 0
            else:
                hr, ndcg, hr_cold, ndcg_cold, hr_warm, ndcg_warm = inference(model, inference_items_dataloader, inference_users_dataloader, dataloader_val, train_for_test_dataloader, device, k=FLAGS.k, margin=FLAGS.margin, lr=FLAGS.lr, num_negatives=FLAGS.num_negatives, num_shards=FLAGS.num_shards, rank=rank)

                if hr > best_hr:
                    best_hr = hr
                    corr_cold_hr = hr_cold
                    corr_warm_hr = hr_warm
                    best_ndcg = ndcg
                    corr_cold_ndcg = ndcg_cold
                    corr_warm_ndcg = ndcg_warm
                    best_epoch = epoch


                if FLAGS.wandb_logging:
                    wandb.log({
                        "Overall HR" : hr,
                        "Cold HR" : hr_cold,
                        "Warm HR" : hr_warm,
                        "Overall NDCG" : ndcg,
                        "Cold NDCG" : ndcg_cold,
                        "Warm NDCG" : ndcg_warm,
                        "Epoch" : epoch,
                        "Best Overall HR" : best_hr,
                        "Best Cold HR" : corr_cold_hr,
                        "Best Warm HR" : corr_warm_hr,
                        "Best Overall NDCG" : best_ndcg,
                        "Best Cold NDCG" : corr_cold_ndcg,
                        "Best Warm NDCG" : corr_warm_ndcg,
                        "Best Epoch" : best_epoch
                    })
                
                print(f"Best HR so far = {best_hr}, Best NDCG so far {best_ndcg}, was at Epoch {best_epoch}")
                print(f"Corresponding (Warm, Cold) Hit Rate = ({corr_warm_hr}, {corr_cold_hr}), Corresponding (Warm. Cold) NDCG = ({corr_warm_ndcg},{corr_cold_ndcg})")

            hr_tensor = torch.tensor(hr, device=device, dtype=torch.float64)
            dist.broadcast(hr_tensor, src=0)
            hr = hr_tensor.item()
            if hr >= best_hr:
                best_hr = hr
                best_model_state_dict = copy.deepcopy(model)

    torch.save(best_model_state_dict, f"{FLAGS.dataset}_M{rank}_model_maml.pt")


def main(argv):
    setproctitle.setproctitle(f"Main ({os.getpid()})")
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed_all(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    torch.backends.cudnn.benchmark = True
    FLAGS.fanouts = list(map(lambda x : int(x), FLAGS.fanouts))
    FLAGS.user_embed_hidden_dims = list(map(lambda x : int(x), FLAGS.user_embed_hidden_dims))
    FLAGS.item_embed_hidden_dims = list(map(lambda x : int(x), FLAGS.item_embed_hidden_dims))

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    device = torch.device(rank) if (torch.cuda.is_available() and FLAGS.device == 'gpu') else torch.device('cpu')

    if rank == 0:
        if FLAGS.wandb_logging:
            wandb.init(project="test", name=FLAGS.run_name, entity="test", config=FLAGS, mode="online")
        user_feature_model, item_feature_model, num_user_feats, num_item_feats, user_feat_counts, item_feat_counts, usable_user_feats, unusable_user_feats, igraph = read_data(FLAGS)
        train_dataloaders, dataloader_val, inference_users_dataloader, inference_items_dataloader, train_for_test_dataloaders  = get_dataloaders(igraph, FLAGS)
        init_param = [user_feature_model, item_feature_model, num_user_feats, num_item_feats, user_feat_counts, item_feat_counts, usable_user_feats, unusable_user_feats, train_dataloaders, train_for_test_dataloaders, dataloader_val, inference_users_dataloader, inference_items_dataloader]
        torch.save(igraph, f"{FLAGS.dataset}_igraph.pt")
    else:
        init_param = [None] * 13
    
    dist.broadcast_object_list(init_param, src=0, device=device)
    train(rank, device, init_param[0], init_param[1], init_param[2], init_param[3], init_param[4], init_param[5], init_param[6], init_param[7], init_param[8][rank], init_param[9][rank], init_param[10], init_param[11], init_param[12])


if __name__ == '__main__':
    app.run(main)