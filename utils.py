from GTTF.custom_funcs import accumulate_one_hop_neighbors, softmax_degree_prob, uniform_degree_prob
from GTTF.framework import gttf
from GTTF.utils import convert_to_compact_adj
from data.datareader import read_ml1m

from data.structs import InteractionGraph, MlInteractionGraph
from data.base import (
    BaseDataset,
    InferenceUsersDataset,
    InferenceItemsDataset,
    InitFn
)

from data.ml1m_dataloader import ML1mCollator, Ml1mInferenceUsersCollator, Ml1mInferenceItemsCollator
from torch.utils.data import DataLoader
from model.ml1m_models import Ml1mItemModel, Ml1mUserModel

def batch_to_device(user_feats, item_feats, neighbor_item_feats, device):
    user_feats_dev = {feat : vals.to(device, non_blocking=True) for feat, vals in user_feats.items()} if user_feats is not None else None
    item_feats_dev = {feat : vals.to(device, non_blocking=True) for feat, vals in item_feats.items()} if item_feats is not None else None
    neighbor_item_feats_dev = {feat : vals.to(device, non_blocking=True) for feat, vals in neighbor_item_feats.items()} if neighbor_item_feats is not None else None

    return user_feats_dev, item_feats_dev, neighbor_item_feats_dev

def get_num_vals(data, key):
    return max([data[id][key] for id in data]) + 1

def read_data(flags):
    usable_user_feats, unusable_user_feats = set(flags.usable_user_feats), set()
    if flags.dataset == 'ml1m':
        user_data, item_data, interactions, num_movie_dates = read_ml1m(datasets_dir=flags.dataset_dir)
        user_feat_counts = {'id' : len(user_data), 'gender' : 2, 'age' : 7, 'occ' : 21}
        item_feat_counts = {'id' : len(item_data), 'date' : num_movie_dates, 'genre' : 18}

        if len(usable_user_feats) == 0:
            usable_user_feats = set(list(user_feat_counts.keys()))
        elif len(usable_user_feats) < len(user_feat_counts):
            unusable_user_feats = set(list(user_feat_counts.keys())) - usable_user_feats

        num_user_feats, num_item_feats = len(user_feat_counts) if flags.user_feat_attn else len(usable_user_feats), len(item_feat_counts) + 1
        user_feature_model, item_feature_model = Ml1mUserModel, Ml1mItemModel
        igraph = MlInteractionGraph(user_data, item_data, interactions, warm_threshold=flags.warm_threshold, mfv_ratio=flags.mfv_ratio, num_shards=flags.num_shards, query_set_length=flags.query_set_length, heuristic_sample_size=flags.heuristic_sample_size, heuristic=flags.heuristic)
        
    assert len(usable_user_feats) > 0
    assert len(usable_user_feats.intersection(unusable_user_feats)) == 0
    assert (len(usable_user_feats) + len(unusable_user_feats)) == len(user_feat_counts)

    usable_user_feats = sorted(list(usable_user_feats))
    unusable_user_feats = sorted(list(unusable_user_feats))

    return user_feature_model, item_feature_model, num_user_feats, num_item_feats, user_feat_counts, item_feat_counts, usable_user_feats, unusable_user_feats, igraph

def get_dataloaders(interaction_graph: InteractionGraph, flags):
    if flags.synthesis_model is not None:
        compact_adj = convert_to_compact_adj(interaction_graph, load_edge_lccsr=flags.load_edge_lccsr, edge_lccsr_path=flags.edge_lccsr_path)
        bias_dist = softmax_degree_prob if flags.sampling_bias else uniform_degree_prob
        bias_func = compact_adj.create_multinomial_sampler(bias_dist, interaction_graph.start_item_id, temp=flags.sampling_softmax_temp, order=flags.order)
        one_hop_runner = gttf(compact_adj, bias_func=bias_func, acc_func=accumulate_one_hop_neighbors)
    else:
        one_hop_runner = None
    
    test_dataset = BaseDataset(interaction_graph, mode='test')
    inf_users_dataset = InferenceUsersDataset(interaction_graph)
    inf_items_dataset = InferenceItemsDataset(interaction_graph)

    if flags.dataset == 'ml1m':
        train_collator = ML1mCollator(interaction_graph, one_hop_runner, mode='train', num_neg_samples=flags.num_negatives, fanouts=flags.fanouts, seed=flags.seed)
        train_for_test_collator = ML1mCollator(interaction_graph, one_hop_runner, mode='train_for_test', num_neg_samples=flags.num_negatives, fanouts=flags.fanouts, seed=flags.seed)
        test_collator = ML1mCollator(interaction_graph, one_hop_runner, mode='test')
        inf_users_collator = Ml1mInferenceUsersCollator(interaction_graph, one_hop_runner, fanouts=flags.fanouts, seed=flags.seed)
        inf_items_collator = Ml1mInferenceItemsCollator(interaction_graph)

    init_fn = InitFn(flags.seed, flags.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=test_collator)
    inf_users_dataloader = DataLoader(inf_users_dataset, batch_size=flags.item_inference_batch_size, collate_fn=inf_users_collator, worker_init_fn=init_fn, pin_memory=True)
    inf_items_dataloader = DataLoader(inf_items_dataset, batch_size=flags.item_inference_batch_size, collate_fn=inf_items_collator, worker_init_fn=init_fn, pin_memory=True)

    train_dataloaders = []
    train_for_test_dataloaders = []
    for i in range(flags.num_shards):
        train_dataset = BaseDataset(interaction_graph, mode='train', shard_num=i)
        train_dataloader = DataLoader(train_dataset, batch_size=flags.batch_size, collate_fn=train_collator, worker_init_fn=init_fn, num_workers=flags.num_workers, shuffle=True, pin_memory=True)
        train_dataloaders.append(train_dataloader)

        train_for_test_dataset = BaseDataset(interaction_graph, mode='train_for_test', shard_num=i)
        train_for_test_dataloader = DataLoader(train_for_test_dataset, batch_size=1, collate_fn=train_for_test_collator, worker_init_fn=init_fn, num_workers=flags.num_workers, shuffle=False, pin_memory=True)
        train_for_test_dataloaders.append(train_for_test_dataloader)
        
    return train_dataloaders, test_dataloader, inf_users_dataloader, inf_items_dataloader, train_for_test_dataloaders