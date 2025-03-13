import numpy as np
import torch
from data.structs import InteractionGraph

from torch.utils.data import DataLoader
from utils import batch_to_device
from tqdm import tqdm
from copy import deepcopy
import torch.distributed as dist

@torch.no_grad()
def compute_a_user_representations(model, user_feats):
    user_feats = {key: tensor[0].unsqueeze(0) for key, tensor in user_feats.items()}
    user_representation = model.user_embedding_model(model.user_feature_model(user_feats))
    
    return user_representation

@torch.no_grad()
def compute_all_user_representations(model, user_loader, device):
    user_representations = torch.zeros((len(user_loader.dataset), model.user_embed_dim), dtype=torch.float32, device=device)
    for _, (user_feats, neighbor_item_feats) in enumerate(user_loader):
        user_feats, _, neighbor_item_feats = batch_to_device(user_feats=user_feats, item_feats=None, neighbor_item_feats=neighbor_item_feats, device=device)
        user_ids = user_feats['id'].long()
        user_feats = model.user_feature_model(user_feats)

        if neighbor_item_feats is not None:
            neighbor_item_feats = model.item_feature_model(neighbor_item_feats)
            user_embeds_precomputed = model.user_embedding_model(user_feats, neighbor_item_feats.view(user_feats.shape[0], -1, neighbor_item_feats.shape[1]))      
        else:
            user_embeds_precomputed = model.user_embedding_model(user_feats)

        user_representations[user_ids, :] = user_embeds_precomputed
    
    return user_representations

@torch.no_grad()
def compute_all_item_representations(model, item_loader, device):
    item_representations = torch.zeros((len(item_loader.dataset), model.user_embed_dim), dtype=torch.float32, device=device)
    for _, item_feats in enumerate(item_loader):
        _, item_feats, _ = batch_to_device(user_feats=None, item_feats=item_feats, neighbor_item_feats=None, device=device)
        item_ids = item_feats['id'] + len(item_loader.dataset.igraph.user_data)

        item_feats = model.item_feature_model(item_feats)
        item_embeddings_precomputed = model.item_embedding_model(item_feats)

        item_ids = item_ids.cpu()
        for i in range(len(item_embeddings_precomputed)):
            item_representations[item_loader.dataset.item_reindexer[item_ids[i].item()], :] = item_embeddings_precomputed[i]

    return item_representations

def compute_ranking_metrics(topk, true_indices):
    membership = (topk == true_indices.reshape(-1, 1)).any(axis=1)
    hitrate_k = 100 * (membership.sum() / membership.shape[0])

    denoms = np.log2(np.argwhere(topk == true_indices.reshape(-1, 1))[:, 1] + 2)
    dcg_k = 1 / denoms
    ndcg_k = 100 * np.sum(dcg_k) / true_indices.shape[0]


    return hitrate_k, ndcg_k

def compute_scores_for_user(model, margin, lr, num_negatives, user_feats, item_feats, user_loader, item_loader, device, i): # this return scores for all items for a single user of size torch.Size([1, 3883]) then to be used to create all scores of size torch.Size([6040, 3883])
    keep_weight = deepcopy(model.state_dict())

    user_feats, item_feats, _ = batch_to_device(user_feats, item_feats, None, device)

    loss_fn = torch.nn.MarginRankingLoss(margin=margin)
    y = torch.tensor([1], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scores = model(user_feats, item_feats).flatten()
    current_batch_size = int(user_feats['id'].shape[0] / ((1 * num_negatives) + 1))
    loss = loss_fn(scores[:current_batch_size].repeat_interleave(1 * num_negatives), scores[current_batch_size:], y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    user_representation = compute_a_user_representations(model, user_feats)
    item_representations = compute_all_item_representations(model, item_loader, device)
    scores = torch.mm(user_representation, item_representations.T)

    model.load_state_dict(keep_weight)

    return scores

def inference(model,
              item_loader : DataLoader,
              user_loader : DataLoader,
              val_loader : DataLoader,
              dataloader_train_for_test : list,
              device : torch.device,
              k=10,
              margin = 1,
              lr = 0.1,
              num_negatives = 1,
              num_shards=8,
              rank=0):
        

    scores_list = []
        
    for i, (user_feats, item_feats) in enumerate(tqdm(dataloader_train_for_test, disable=True)): # this includes negatives and other shard query sets
        scores_list.append(compute_scores_for_user(model, margin, lr, num_negatives, user_feats, item_feats, user_loader, item_loader, device, i))

    with torch.no_grad():
        all_scores = torch.cat(scores_list)
        dist.reduce(all_scores, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            all_scores = torch.div(all_scores, num_shards)
            iGraph : InteractionGraph = val_loader.dataset.igraph
            test_edges, training_edges = next(iter(val_loader))

            test_users = test_edges[:, 0]
            test_items = test_edges[:, 1]

            tail_idxs = iGraph.is_cold[test_items].nonzero()[0]
            head_idxs = (~iGraph.is_cold[test_items]).nonzero()[0]

            train_users = training_edges[0]
            train_items = training_edges[1].apply_(lambda x : iGraph.item_reindexer[x])

            all_scores[train_users, train_items] = -float("inf")
            _, topk_items = torch.topk(all_scores, k=k, largest=True, dim=-1)

            topk = topk_items.cpu().numpy()
            true_items = test_items.apply_(lambda x : iGraph.item_reindexer[x]).numpy()

            hitrate_k_combined, ndcg_k_combined = compute_ranking_metrics(topk=topk[test_users], true_indices=true_items)
            hitrate_k_cold, ndcg_k_cold = compute_ranking_metrics(topk=topk[test_users[tail_idxs]], true_indices=true_items[tail_idxs])
            hitrate_k_warm, ndcg_k_warm = compute_ranking_metrics(topk=topk[test_users[head_idxs]], true_indices=true_items[head_idxs])

            print(f"HR@{k} Overall = {hitrate_k_combined}, NDCG@{k} Overall = {ndcg_k_combined}")
            print(f"HR@{k} Cold = {hitrate_k_cold}, NDCG@{k} Cold = {ndcg_k_cold}")
            print(f"HR@{k} Warm = {hitrate_k_warm}, NDCG@{k} Warm = {ndcg_k_warm}")

            return hitrate_k_combined, ndcg_k_combined, hitrate_k_cold, ndcg_k_cold, hitrate_k_warm, ndcg_k_warm