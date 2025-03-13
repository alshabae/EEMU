import torch
import torch.nn as nn

from model.common import UnusableFeaturePredictor

class Ml1mUserModel(nn.Module):
    def __init__(self,
        user_feat_counts,
        usable_feats,
        unusable_feats,
        feat_embed_dim=64,
        feat_attention=False
    ):
        super(Ml1mUserModel, self).__init__()
        self.feat_dict = nn.ModuleDict({feat_name : nn.Embedding(user_feat_counts[feat_name], embedding_dim=feat_embed_dim) for feat_name in user_feat_counts})
        self.usable_feats = usable_feats
        self.unusable_feat = unusable_feats
        self.num_usable_feats = len(usable_feats)
        self.num_unusable_feats = len(unusable_feats)
        self.feat_attention = feat_attention
        
        for feat in usable_feats:
            assert feat in self.feat_dict
        
        if self.num_unusable_feats > 0:
            for feat in unusable_feats:
                assert feat in self.feat_dict

            if self.feat_attention:
                self.num_cats_per_unusable_feat = {feat: self.feat_dict[feat].weight.shape[0] for feat in unusable_feats}
                self.unknown_feat_predictor = UnusableFeaturePredictor(feat_embed_dim, num_usable_feats=self.num_usable_feats, num_cats_per_unusable_feat=self.num_cats_per_unusable_feat)
            

    def forward(self, user_feats):
        if self.num_unusable_feats == 0:
            combined_rep = torch.cat([self.feat_dict[feat_name](user_feats[feat_name]) for feat_name in self.feat_dict], dim=1)
        else:
            usable_feat_list = [self.feat_dict[feat_name](user_feats[feat_name]) for feat_name in self.usable_feats]
            usable_feat_embeddings = torch.cat(usable_feat_list, dim=1)
            if self.feat_attention:
                usable_feat_seq = torch.stack(usable_feat_list, dim=1)
                ufps = self.unknown_feat_predictor(usable_feat_seq)
                unusable_feat_embeddings = torch.cat([torch.matmul(ufps[feat_name], self.feat_dict[feat_name].weight) for feat_name in ufps], dim=1)
                combined_rep = torch.cat([usable_feat_embeddings, unusable_feat_embeddings], dim=1)
            else:
                combined_rep = usable_feat_embeddings
        
        return combined_rep

class Ml1mItemModel(nn.Module):
    def __init__(self,
        item_feat_counts,
        feat_embed_dim=64,
        **kwargs
    ):
        super(Ml1mItemModel, self).__init__()
        self.id_embeddings = nn.Embedding(item_feat_counts['id'], feat_embed_dim)
        self.date_embeddings = nn.Embedding(item_feat_counts['date'], feat_embed_dim)
        self.genre_embedding_matrix = nn.Parameter(torch.empty(item_feat_counts['genre'], feat_embed_dim))
        nn.init.xavier_normal_(self.genre_embedding_matrix)
        self.dense_transform = nn.Linear(kwargs['dense_feat_input_dim'], feat_embed_dim)

        self.register_parameter("genre_embedding_matrix", self.genre_embedding_matrix)

    def forward(self, item_feats):
        id_embeddings = self.id_embeddings(item_feats['id'])
        date_embeddings = self.date_embeddings(item_feats['date'])       
        genre_embeddings = torch.matmul(item_feats['genres'], self.genre_embedding_matrix)
        dense_embeddings = self.dense_transform(item_feats['title_embedding'])

        combined_rep = torch.cat([id_embeddings, date_embeddings, genre_embeddings, dense_embeddings], dim=1)
        return combined_rep