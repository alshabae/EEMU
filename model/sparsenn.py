import torch.nn as nn
from model.synthesis_models import (
    MeanSynthesis,
    UserAttentionSynthesis,
    RecurrentSynthesis
)

from model.common import (
    DotProduct,
    MLPTower
)

class SparseNN(nn.Module):
    def __init__(self,
        user_feature_model,
        item_feature_model,
        num_user_feats,
        num_item_feats,
        user_feat_counts,
        item_feat_counts,
        usable_user_feats,
        unusable_user_feats,
        user_feat_attn=False,
        feat_embed_dim=96,
        dense_feat_embed_dim=384,
        user_embed_hidden_dims=[512, 256, 128],
        item_embed_hidden_dims=[256, 128],
        projection_dim=64,
        user_embed_dim=128,
        seq_len=15,
        recurr_hidden_size=32,
        recurr_num_layers=2,
        nhead=1,
        synthesis_model='mean',
        recurr_mode='gru',
    ):
        super(SparseNN, self).__init__()
        self.feat_embed_dim = feat_embed_dim
        self.user_embed_hidden_dims = user_embed_hidden_dims
        self.item_embed_hidden_dims = item_embed_hidden_dims
        self.user_embed_dim = user_embed_dim

        self.user_feature_model = user_feature_model(user_feat_counts, usable_user_feats, unusable_user_feats, feat_embed_dim=feat_embed_dim, feat_attention=user_feat_attn)
        self.item_feature_model = item_feature_model(item_feat_counts, feat_embed_dim=feat_embed_dim, dense_feat_input_dim=dense_feat_embed_dim)
        user_feat_dim = num_user_feats * feat_embed_dim
        item_feat_dim = num_item_feats * feat_embed_dim

        if synthesis_model == 'mean':
            self.user_embedding_model = MeanSynthesis(user_feat_dim, item_feat_dim, user_embed_dim=user_embed_dim, hidden_dims=user_embed_hidden_dims)
        elif synthesis_model == 'attn':
            self.user_embedding_model = UserAttentionSynthesis(
                user_feat_dim, 
                item_feat_dim, 
                projection_dim=projection_dim, 
                user_embed_dim=user_embed_dim, 
                hidden_dims=user_embed_hidden_dims,
                num_heads=nhead
            )
        elif synthesis_model == 'recurr':
            self.user_embedding_model = RecurrentSynthesis(
                seq_len, 
                user_feat_dim, 
                item_feat_dim, 
                recurr_hidden_size, 
                recurr_num_layers, 
                user_embed_dim=user_embed_dim, 
                hidden_dims=user_embed_hidden_dims,
                mode=recurr_mode
            )
        elif synthesis_model is None:
            self.user_embedding_model = MLPTower(user_feat_dim, user_embed_hidden_dims, user_embed_dim)
        else:
            raise NotImplementedError

        self.item_embedding_model = MLPTower(item_feat_dim, item_embed_hidden_dims, user_embed_dim)
        self.scoring_model = DotProduct()
        
    def forward(self, 
        user_feats = None, 
        item_feats = None, 
        neighbor_item_feats = None,
        user_embeddings_precomputed = None,
        item_embeddings_precomputed = None,
    ):
        if user_embeddings_precomputed is None:
            user_feats = self.user_feature_model(user_feats)
            if neighbor_item_feats is not None: # Sampling model
                neighbor_item_feats = self.item_feature_model(neighbor_item_feats)
                user_embeds = self.user_embedding_model(user_feats, neighbor_item_feats.view(user_feats.shape[0], -1, neighbor_item_feats.shape[1]))
            else: # Base Model
                user_embeds = self.user_embedding_model(user_feats)
        else:
            user_embeds = user_embeddings_precomputed

        if item_embeddings_precomputed is None:
            item_feats = self.item_feature_model(item_feats)
            item_embeds = self.item_embedding_model(item_feats)
        else:
            item_embeds = item_embeddings_precomputed

        scores = self.scoring_model(user_embeds, item_embeds)
        return scores