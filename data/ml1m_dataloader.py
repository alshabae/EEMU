from typing import Callable, List, Optional

import numpy as np
import torch
from data.base import BaseCollator
from data.structs import InteractionGraph


class ML1mCollator(BaseCollator):
    def __init__(self, 
                 igraph: InteractionGraph, 
                 runner: Callable[[np.ndarray, List[int], Optional[np.ndarray], Optional[bool]], np.ndarray], 
                 mode: str, 
                 num_neg_samples=1, 
                 fanouts=None, 
                 seed=0) -> None:
        super().__init__(igraph, runner, mode, num_neg_samples, fanouts, seed)
        
    def _fetch_user_data(self, users):
        user_feats = np.array([(user_id, self.user_data[user_id]['gender'], self.user_data[user_id]['age'], self.user_data[user_id]['occupation']) for user_id in users])
        user_ids = torch.as_tensor(user_feats[:, 0], dtype=torch.int64)
        user_genders = torch.as_tensor(user_feats[:, 1], dtype=torch.int64)
        user_ages = torch.as_tensor(user_feats[:, 2], dtype=torch.int64)
        user_occs = torch.as_tensor(user_feats[:, 3], dtype=torch.int64)

        return {'id' : user_ids, 'gender' : user_genders, 'age' : user_ages, 'occ' : user_occs}

    def _fetch_item_data(self, items):
        item_dates = torch.as_tensor(np.array([self.item_data[movie_id]['date'] for movie_id in items]), dtype=torch.int64)
        item_genres = torch.as_tensor(np.array([self.item_data[movie_id]['genres'] for movie_id in items]), dtype=torch.float32)
        item_title_embeddings = torch.as_tensor(np.array([self.item_data[movie_id]['title_embedding'] for movie_id in items]), dtype=torch.float32)
        item_ids = torch.as_tensor(items, dtype=torch.int64) - len(self.user_data)

        return {'id' : item_ids, 'date' : item_dates, 'genres' : item_genres, 'title_embedding' : item_title_embeddings}

class Ml1mInferenceItemsCollator(ML1mCollator):
    def __init__(self, igraph: InteractionGraph) -> None:
        super().__init__(igraph, None, None, None, None, None)

    def __call__(self, batch):
        batch = np.array(batch)
        return self._fetch_item_data(batch)

class Ml1mInferenceUsersCollator(ML1mCollator):
    def __init__(self, 
                 igraph: InteractionGraph, 
                 runner: Callable[[np.ndarray, List[int], Optional[np.ndarray], Optional[bool]], np.ndarray], 
                 mode: str = 'test', 
                 num_neg_samples=1, 
                 fanouts=None, 
                 seed=0) -> None:
        super().__init__(igraph, runner, mode, num_neg_samples, fanouts, seed)
    
    def __call__(self, batch):
        batch = np.array(batch)
        user_feats = self._fetch_user_data(batch)
        one_hop_item_feats = self._fetch_neighbor_item_data(batch, edge_items=None, filter=False)
        return user_feats, one_hop_item_feats