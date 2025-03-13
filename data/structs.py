from random import random
from tqdm import tqdm
import numpy as np
import scipy.sparse

class InteractionGraph:
    def __init__(self, user_data, item_data, interactions) -> None:
        self.user_data = user_data
        self.item_data = item_data
        self.interactions = interactions
        self.train_edges, self.validation_edges, self.test_edges = [], [], []
        self.adj_matrix: scipy.sparse.dok_matrix = None

        self.all_item_ids = sorted(list(self.item_data.keys()))
        self.item_reindexer = {}
        for item_id in self.all_item_ids:
            self.item_reindexer[item_id] = len(self.item_reindexer)

        self.reverse_item_indexer = {v : k for k, v in self.item_reindexer.items()}

    def split_statistics(self):
        training_items = set(self.train_edges[:, 1])
        validation_items = set(self.validation_edges[:, 1])
        test_items = set(self.test_edges[:, 1])

        print("Total number of training edges = {}".format(len(self.train_edges)))
        print("Total number of items = {}".format(len(self.item_data)))
        print("Total number of users = {}".format(len(self.user_data)))
        print("Number of items present across training edges = {}".format(len(training_items)))
        print("Number of items present across val edges = {}".format(len(validation_items)))
        print("Number of items present across test edges = {}".format(len(test_items)))
        print("Average item degree = {}".format(np.mean(self.item_degrees)))
        print("Average user degree = {}".format(np.mean(self.user_degrees)))
        print("min item degree = {}".format(np.min(self.item_degrees)))
        print("min user degree = {}".format(np.min(self.user_degrees)))
        print("max item degree = {}".format(np.max(self.item_degrees)))
        print("max user degree = {}".format(np.max(self.user_degrees)))

        train_val_common_items = training_items.intersection(validation_items)
        train_test_common_items = training_items.intersection(test_items)

        print('Number of items common between train and validation edges = {}'.format(len(train_val_common_items)))
        print('Number of items common between train and test edges = {}'.format(len(train_test_common_items)))

        validation_items = np.array(list(validation_items))
        test_items = np.array(list(test_items))

        num_cold_items_in_val = np.sum(self.is_cold[validation_items])
        num_cold_items_in_test = np.sum(self.is_cold[test_items])

        print('Number of cold items in validation set = {}'.format(num_cold_items_in_val))
        print('Number of cold items in test set = {}'.format(num_cold_items_in_test))


    def create_bipartite_graph(self):
        num_nodes = len(self.user_data) + len(self.item_data) # Num users + num items 
        self.adj_matrix = scipy.sparse.dok_matrix((num_nodes, num_nodes), dtype=bool)  # TODO: Maybe we can optimize with lower precision data types
        
        for edge in self.train_edges:
            self.adj_matrix[edge[0], edge[1]] = 1
            self.adj_matrix[edge[1], edge[0]] = 1

        self.adj_matrix = self.adj_matrix.tocsr()
    
    def compute_tail_distribution(self, warm_threshold):
        self.is_cold = np.zeros((self.adj_matrix.shape[0]), dtype=bool)
        self.start_item_id = len(self.user_data)
        self.item_degrees = np.array(self.adj_matrix[self.start_item_id:].sum(axis=1)).flatten()
        self.user_degrees = np.array(self.adj_matrix[:self.start_item_id].sum(axis=1)).flatten()

        cold_items = np.argsort(self.item_degrees)[:int((1 - warm_threshold) * len(self.item_degrees))] + self.start_item_id
        self.is_cold[cold_items] = True

    def __getitem__(self, user_id):
        assert user_id < len(self.user_data), "User ID out of bounds"
        assert isinstance(self.adj_matrix, scipy.sparse.csr_matrix), "Bipartite graph not created: must call create_bipartite_graph first"
        return np.array(self.adj_matrix[user_id, self.start_item_id:].todense()).flatten().nonzero()[0] + self.start_item_id

    def create_data_split(self):
        raise NotImplementedError()
    

class MlInteractionGraph(InteractionGraph):
    def __init__(self, user_data, item_data, interactions, warm_threshold=0.2, mfv_ratio=0.0, num_shards=10, query_set_length=3, heuristic_sample_size=1, heuristic="other_shards_query") -> None:
        super().__init__(user_data, item_data, interactions)
        self.create_data_split()
        self.create_bipartite_graph()
        assert (warm_threshold < 1.0 and warm_threshold > 0.0)
        self.warm_threshold = warm_threshold
        self.compute_tail_distribution()
        self.split_statistics()

        self.num_shards = num_shards
        self.query_set_length = query_set_length
        self.heuristic_sample_size = heuristic_sample_size
        self.heuristic = heuristic
        self.shard_train_edges_per_user((self.num_shards))

        self.mfv_ratio = mfv_ratio
        if self.mfv_ratio > 0.0:
            self.hide_user_features()
    
    def hide_user_features(self):
        user_ids = list(self.user_data.keys())
        random.shuffle(user_ids)
        mfv_user_ids = user_ids[:int(self.mfv_ratio) * len(user_ids)]
        for user_id in mfv_user_ids:
            hide_gender = random.randint(0, 1)
            hide_age = random.randint(0, 1)
            hide_occ = random.randint(0, 1)

            if hide_gender:
                self.user_data[user_id]['gender'] = -1

            if hide_age:
                self.user_data[user_id]['age'] = -1

            if hide_occ:
                self.user_data[user_id]['occupation'] = -1
    
    def create_data_split(self):
        # Leave one out validation - for each user the latest interaction is a test item and the second latest item is the validation item
        print('Creating data split')
        self.all_edges = set()
        self.interaction_time_stamps = {}
        for user_id in tqdm(self.interactions):
            sorted_interactions = sorted(self.interactions[user_id], key=lambda x : x[2])
            test_edge = [user_id, sorted_interactions[-1][0]]
            val_edge = [user_id, sorted_interactions[-2][0]]
            self.all_edges.add((user_id, sorted_interactions[-2][0]))

            train_edges = [[user_id, interaction[0]] for interaction in sorted_interactions[:-2]]
            for interaction in sorted_interactions[:-2]:
                self.all_edges.add((user_id, interaction[0]))
                self.interaction_time_stamps[(user_id, interaction[0])] = interaction[2]
                self.interaction_time_stamps[(interaction[0], user_id)] = interaction[2]             

            self.train_edges += train_edges
            self.validation_edges.append(val_edge)
            self.test_edges.append(test_edge)
        
        self.train_edges = np.array(self.train_edges)
        self.validation_edges = np.array(self.validation_edges)
        self.test_edges = np.array(self.test_edges)
    
    def create_data_split_v2(self):
        print('Creating data split')
        self.all_edges = set()
        self.interaction_time_stamps = {}
        for user_id in tqdm(self.interactions):
            sorted_interactions = sorted(self.interactions[user_id], key=lambda x : x[2])
            num_test_interactions = int(0.1 * len(sorted_interactions))

            for interaction in sorted_interactions[:-num_test_interactions]:
                self.all_edges.add((user_id, interaction[0]))
                self.train_edges.append([user_id, interaction[0]])
                self.interaction_time_stamps[(user_id, interaction[0])] = interaction[2]
                self.interaction_time_stamps[(interaction[0], user_id)] = interaction[2]
            
            for interaction in sorted_interactions[-num_test_interactions:]:
                self.test_edges.append([user_id, interaction[0]])

        
        self.train_edges = np.array(self.train_edges)
        self.validation_edges = np.array(self.validation_edges)
        self.test_edges = np.array(self.test_edges)

    
    def compute_tail_distribution(self):
        return super().compute_tail_distribution(self.warm_threshold)
    
    def shard_train_edges_per_user(self, num_shards):
        self.edges_per_user = {}
        for k, v in self.train_edges:
            self.edges_per_user.setdefault(k, []).append((k,v))

        self.shards_per_user = {}
        for k in self.edges_per_user.keys():
            user_edges = np.array(self.edges_per_user[k])
            np.random.shuffle(user_edges)
            user_shards = np.array_split(user_edges, num_shards)
            user_shards_support_query = [[shard[:-self.query_set_length], shard[-self.query_set_length:]] for shard in user_shards]
            self.shards_per_user[k] = user_shards_support_query
    
