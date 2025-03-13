from typing import Any, Callable, List, Optional, Tuple
import scipy.sparse
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import setproctitle
from data.structs import InteractionGraph
import os

class InitFn:
    def __init__(self, seed, num_workers) -> None:
        self.seed = seed
        self.num_workers = num_workers
    
    def __call__(self, worker_id):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        setproctitle.setproctitle(f"Dataloader Process {worker_id} ({os.getpid()})")


class BaseDataset(Dataset):
    def __init__(self, igraph : InteractionGraph, mode='train', shard_num=0) -> None:
        self.igraph = igraph
        self.user_data = igraph.user_data
        self.item_data = igraph.item_data
        self.shards_per_user = igraph.shards_per_user
        self.mode = mode
        self.shard_num = shard_num
        self.other_shard_nums = [i for i in range(len(self.shards_per_user[0])) if i != self.shard_num]
        
        if mode == 'train':
            self.edges = self.shards_per_user # redundunt
        elif mode == 'val':
            self.edges = igraph.validation_edges
        elif mode == 'train_for_test':
            self.edges = self.shards_per_user # redundunt
        else:
            self.edges = igraph.test_edges

    def __len__(self):
        return len(self.user_data)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            return self.shards_per_user[index][self.shard_num]
        elif self.mode == 'train_for_test' and self.igraph.heuristic =='all_biased_sample':
            other_shards = []
            for shard in range(len(self.shards_per_user[0])):
                other_shards = other_shards + self.shards_per_user[index][shard][0].tolist() + self.shards_per_user[index][shard][1].tolist()
            return other_shards
        else:
            return self.edges[index]

class InferenceItemsDataset(Dataset):
    def __init__(self, igraph : InteractionGraph) -> None:
        self.igraph = igraph
        self.all_item_ids = igraph.all_item_ids
        self.item_reindexer = igraph.item_reindexer

    def __len__(self):
        return len(self.all_item_ids)
    
    def __getitem__(self, index):
        return self.all_item_ids[index]

class InferenceUsersDataset(Dataset):
    def __init__(self, igraph : InteractionGraph) -> None:
        self.all_user_ids = sorted(list(igraph.user_data.keys()))
    
    def __len__(self):
        return len(self.all_user_ids)
    
    def __getitem__(self, index):
        return self.all_user_ids[index]

class BaseCollator:
    def __init__(self, 
                 igraph: InteractionGraph, 
                 runner: Callable[[np.ndarray, List[int], Optional[np.ndarray], Optional[bool]], np.ndarray],
                 mode: str, 
                 num_neg_samples=1, 
                 fanouts=None,
                 seed=0) -> None:
        self.igraph = igraph
        self.runner = runner
        self.fanouts = fanouts
        self.user_data = igraph.user_data
        self.item_data = igraph.item_data
        self.adj_matrix = igraph.adj_matrix
        self.num_neg_samples = num_neg_samples
        self.mode = mode
        self.rng = np.random.Generator(np.random.PCG64(seed=seed))
    
    def _generate_in_and_oob_negatives(self, positive_edges):
        item_start_id = len(self.user_data)
        pos_edges = np.array(positive_edges)

        negative_edges = []
        for i, (user_id, _) in enumerate(positive_edges):

            # Out of batch negative Sampling
            candidate_item_probs = np.asarray(self.adj_matrix[user_id, item_start_id:].todense()).flatten() == 0
            candidate_item_probs = candidate_item_probs / candidate_item_probs.sum()

            valid_samples = False
            while not valid_samples:
                neg_items = np.random.choice(candidate_item_probs.shape[0], (self.num_neg_samples,), p=candidate_item_probs)     
                for neg_item in neg_items:
                    if (user_id, neg_item + item_start_id) in self.igraph.all_edges:
                        valid_samples = False
                        break

                    valid_samples = True
            
            # In batch negative sampling
            in_batch_candidates = np.concatenate((pos_edges[:i], pos_edges[(i+1):]))[:, 1]
            idxs_to_delete = [idx for idx, candidate in enumerate(in_batch_candidates) if (user_id, candidate) in self.igraph.all_edges]
            valid_candidates = np.delete(in_batch_candidates, idxs_to_delete)
            in_batch_negs = self.rng.choice(valid_candidates, (self.num_neg_samples,), replace=False)

            for neg_item in neg_items:
                negative_edges.append([user_id, neg_item + item_start_id])

            for neg_item in in_batch_negs:
                negative_edges.append([user_id, neg_item])
        
        return negative_edges
    
    def get_edges_in_batch_of_users(self, batch_of_users):
        edges = []
        for user in batch_of_users:
            edges.extend(user[0]) # get support edges
            edges.extend(user[1]) # get query edges
        return np.array(edges)
        
    def _generate_negatives(self, positive_edges): # generate negative then append them to the original list after support and query
        item_start_id = len(self.user_data)
        batch_edges = self.get_edges_in_batch_of_users(positive_edges)
        positives_and_negatives = []
        for i, (support, query) in enumerate(positive_edges):
            if len(support) > 0:
                user_id = support[0][0]
                num_negatives = self.num_neg_samples * (len(support) + len(query))
                num_support_negatives = self.num_neg_samples * len(support)
                num_query_negatives = self.num_neg_samples * len(query)
                negative_edges = []
                
                # Out of batch negative Sampling
                candidate_item_probs = np.asarray(self.adj_matrix[user_id, item_start_id:].todense()).flatten() == 0
                candidate_item_probs = candidate_item_probs / candidate_item_probs.sum()

                valid_samples = False
                while not valid_samples:
                    neg_items = np.random.choice(candidate_item_probs.shape[0], (num_negatives,), p=candidate_item_probs)     
                    for neg_item in neg_items:
                        if (user_id, neg_item + item_start_id) in self.igraph.all_edges:
                            valid_samples = False
                            break

                        valid_samples = True
                
                # In batch negative sampling

                in_batch_candidates = batch_edges[batch_edges[:, 0] != user_id][:, 1] 
                idxs_to_delete = [idx for idx, candidate in enumerate(in_batch_candidates) if (user_id, candidate) in self.igraph.all_edges]
                valid_candidates = np.delete(in_batch_candidates, idxs_to_delete)      
                in_batch_negs = self.rng.choice(valid_candidates, (num_negatives,), replace=(False if num_negatives<=len(valid_candidates) else True))

                for neg_item in neg_items:
                    negative_edges.append([user_id, neg_item + item_start_id])

                for neg_item in in_batch_negs:
                    negative_edges.append([user_id, neg_item])

                support_and_negatives = np.concatenate((self.rng.permutation(support), np.array(negative_edges[:num_support_negatives]), np.array(negative_edges[num_negatives:num_negatives+num_support_negatives])))
                query_and_negatives = np.concatenate((self.rng.permutation(query), np.array(negative_edges[num_support_negatives:num_negatives]), np.array(negative_edges[num_negatives+num_support_negatives:])))
                positives_and_negatives.append([support_and_negatives, query_and_negatives])
            else: # if support is empty
                user_id = query[0][0]
                num_negatives = self.num_neg_samples * (len(query))
                num_query_negatives = self.num_neg_samples * len(query)
                negative_edges = []
                
                # Out of batch negative Sampling
                candidate_item_probs = np.asarray(self.adj_matrix[user_id, item_start_id:].todense()).flatten() == 0
                candidate_item_probs = candidate_item_probs / candidate_item_probs.sum()

                valid_samples = False
                while not valid_samples:
                    neg_items = np.random.choice(candidate_item_probs.shape[0], (num_negatives,), p=candidate_item_probs)     
                    for neg_item in neg_items:
                        if (user_id, neg_item + item_start_id) in self.igraph.all_edges:
                            valid_samples = False
                            break

                        valid_samples = True
                
                # In batch negative sampling

                in_batch_candidates = batch_edges[batch_edges[:, 0] != user_id][:, 1] 
                idxs_to_delete = [idx for idx, candidate in enumerate(in_batch_candidates) if (user_id, candidate) in self.igraph.all_edges]
                valid_candidates = np.delete(in_batch_candidates, idxs_to_delete)      
                in_batch_negs = self.rng.choice(valid_candidates, (num_negatives,), replace=(False if num_negatives<=len(valid_candidates) else True))

                for neg_item in neg_items:
                    negative_edges.append([user_id, neg_item + item_start_id])

                for neg_item in in_batch_negs:
                    negative_edges.append([user_id, neg_item])

                query_and_negatives = np.concatenate((self.rng.permutation(query), np.array(negative_edges)))
                positives_and_negatives.append([None, query_and_negatives])

        return positives_and_negatives

    def _generate_negatives_for_testing(self, positive_edges): # generate negative then append them to the original of other query shards, out of batch only
        item_start_id = len(self.user_data)
        negative_edges = []
        num_negatives = self.num_neg_samples * (len(positive_edges))
        user_id = positive_edges[0][0]

        # Out of batch negative Sampling
        candidate_item_probs = np.asarray(self.adj_matrix[user_id, item_start_id:].todense()).flatten() == 0
        candidate_item_probs = candidate_item_probs / candidate_item_probs.sum()

        valid_samples = False
        while not valid_samples:
            neg_items = np.random.choice(candidate_item_probs.shape[0], (num_negatives,), p=candidate_item_probs)     
            for neg_item in neg_items:
                if (user_id, neg_item + item_start_id) in self.igraph.all_edges:
                    valid_samples = False
                    break

                valid_samples = True

        for neg_item in neg_items:
            negative_edges.append([user_id, neg_item + item_start_id])

        return np.array(negative_edges)
    
    def _generate_negatives_out_of_batch(self, positive_edges):
        item_start_id = len(self.user_data)
        batch_edges = self.get_edges_in_batch_of_users(positive_edges)
        positives_and_negatives = []
        for i, (support, query) in enumerate(positive_edges):
            if len(support) > 0:
                user_id = support[0][0]
                num_negatives = self.num_neg_samples * (len(support) + len(query))
                num_support_negatives = self.num_neg_samples * len(support)
                num_query_negatives = self.num_neg_samples * len(query)
                negative_edges = []
                
                # Out of batch negative Sampling
                candidate_item_probs = np.asarray(self.adj_matrix[user_id, item_start_id:].todense()).flatten() == 0
                candidate_item_probs = candidate_item_probs / candidate_item_probs.sum()

                valid_samples = False
                while not valid_samples:
                    neg_items = np.random.choice(candidate_item_probs.shape[0], (num_negatives,), p=candidate_item_probs)     
                    for neg_item in neg_items:
                        if (user_id, neg_item + item_start_id) in self.igraph.all_edges:
                            valid_samples = False
                            break

                        valid_samples = True

                for neg_item in neg_items:
                    negative_edges.append([user_id, neg_item + item_start_id])


                support_and_negatives = np.concatenate((self.rng.permutation(support), np.array(negative_edges[:num_support_negatives])))
                query_and_negatives = np.concatenate((self.rng.permutation(query), np.array(negative_edges[num_support_negatives:num_negatives])))
                positives_and_negatives.append([support_and_negatives, query_and_negatives])
            else: # if support is empty
                user_id = query[0][0]
                num_negatives = self.num_neg_samples * (len(query))
                num_query_negatives = self.num_neg_samples * len(query)
                negative_edges = []
                
                # Out of batch negative Sampling
                candidate_item_probs = np.asarray(self.adj_matrix[user_id, item_start_id:].todense()).flatten() == 0
                candidate_item_probs = candidate_item_probs / candidate_item_probs.sum()

                valid_samples = False
                while not valid_samples:
                    neg_items = np.random.choice(candidate_item_probs.shape[0], (num_negatives,), p=candidate_item_probs)     
                    for neg_item in neg_items:
                        if (user_id, neg_item + item_start_id) in self.igraph.all_edges:
                            valid_samples = False
                            break

                        valid_samples = True

                for neg_item in neg_items:
                    negative_edges.append([user_id, neg_item + item_start_id])

                query_and_negatives = np.concatenate((self.rng.permutation(query), np.array(negative_edges)))
                positives_and_negatives.append([None, query_and_negatives])

        return positives_and_negatives

    def _fetch_user_data(self, users) -> Tuple[torch.Tensor,...]:
        raise NotImplementedError
    
    def _fetch_item_data(self, items) -> Tuple[torch.Tensor,...]:
        raise NotImplementedError
    
    def _fetch_neighbor_item_data(self, users, edge_items=None, filter=False):
        if self.runner:
            one_hop_edges = self.runner(users, self.fanouts, edge_items=edge_items, filter=filter)
            unique_one_hop_items, unique_one_hop_idxs = np.unique(one_hop_edges[:, 1], return_inverse=True)
            unique_one_hop_idxs = torch.as_tensor(unique_one_hop_idxs)
            one_hop_ids, one_hop_dates, one_hop_genres, one_hop_title_embeddings = self._fetch_item_data(unique_one_hop_items)
            one_hop_item_feats = one_hop_ids[unique_one_hop_idxs], one_hop_dates[unique_one_hop_idxs], one_hop_genres[unique_one_hop_idxs], one_hop_title_embeddings[unique_one_hop_idxs]
        else:
            one_hop_item_feats = None
        
        return one_hop_item_feats
    
    def _select_from_positives(self, positive_edges, hurestic, temp=0.01):
        if hurestic == 'all_biased_sample':
            edges_array = np.array(positive_edges[0])
            items = edges_array[:,1]
            org_items = items - self.igraph.start_item_id
            item_degres = self.igraph.item_degrees[org_items]
            inverse_degrees = 1 / item_degres
            inverse_degrees = inverse_degrees - inverse_degrees.max()
            probs = np.exp(inverse_degrees / temp)
            softmax = probs / probs.sum()
            biased_sample = edges_array[np.random.choice(edges_array.shape[0], self.igraph.heuristic_sample_size, p=softmax, replace=True)] #False
            return biased_sample
        
    def __call__(self, positive_edges):
        if self.mode == 'val' or self.mode == 'test':
            true_edges = np.array(positive_edges)
            training_edges = scipy.sparse.find(self.adj_matrix[:self.igraph.start_item_id, :])
            training_edges = (torch.as_tensor(training_edges[0], dtype=torch.int64), torch.as_tensor(training_edges[1], dtype=torch.int64))
            return torch.as_tensor(true_edges, dtype=torch.int64), training_edges
        
        elif self.mode == 'train_for_test':
            if(self.num_neg_samples>0):
                selected_positive_edges = self._select_from_positives(positive_edges, self.igraph.heuristic)
                negatives = self._generate_negatives_for_testing(selected_positive_edges)
                edges = np.concatenate((selected_positive_edges, negatives), axis=0)
            else:
                edges = np.array(positive_edges[0])

            user_feats = self._fetch_user_data(edges[:, 0])
            item_feats = self._fetch_item_data(edges[:, 1])
            
            return user_feats, item_feats
        else:
            if(self.num_neg_samples>0):
                edges = self._generate_negatives(positive_edges)
            else:
                edges = positive_edges
            
            support_user_feats = [self._fetch_user_data(edges[i][0][:, 0]) if edges[i][0] is not None else None for i in range(len(edges))] # pass None if there is no support set
            query_user_feats = [self._fetch_user_data(edges[i][1][:, 0]) for i in range(len(edges))]
            support_item_feats = [self._fetch_item_data(edges[i][0][:, 1]) if edges[i][0] is not None else None for i in range(len(edges))] # pass None if there is no support set
            query_item_feats = [self._fetch_item_data(edges[i][1][:, 1]) for i in range(len(edges))]

            return support_user_feats, query_user_feats, support_item_feats, query_item_feats



        



        
