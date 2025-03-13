from typing import List
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dims: List[int], add_bias=True, act="gelu", apply_layernorm=False, elemwise_affine=False):
        super().__init__()
        self._activation = self._get_activation(act)
        self._apply_layernorm = apply_layernorm
        self._elemwise_affine = elemwise_affine
        self._add_bias = add_bias
        self._model = self._create_model(dims)

    def _create_model(self, dims):
        layers = nn.ModuleList()
        for i in range(1, len(dims)):
            layer = nn.Linear(dims[i-1], dims[i]) if self._add_bias else nn.Linear(dims[i-1], dims[i], bias=False)
            layers.append(layer)

            if i < len(dims) - 1:
                if self._apply_layernorm:
                    layers.append(nn.LayerNorm(dims[i], elementwise_affine=self._elemwise_affine))

                layers.append(self._activation)
        
        return nn.Sequential(*layers)

    def _get_activation(self, act):
        if act == 'gelu':
            return nn.GELU()
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'mish':
            return nn.Mish()
        elif act == 'tanh':
            return nn.Tanh()
        else:
            raise NotImplementedError


    def forward(self, input):
        return self._model(input)
    

class MLPTower(nn.Module):
    def __init__(self, feat_dim: int, hidden_dims: List[int], embed_dim: int, act='gelu'):
        super(MLPTower, self).__init__()
        self.dims = [feat_dim] + hidden_dims + [embed_dim]
        self.mlp = MLP(self.dims, apply_layernorm=True, act=act, elemwise_affine=True)
    
    def forward(self, input):
        return self.mlp(input)

class DotCompress(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], act='gelu'):
        super(DotCompress, self).__init__()
        self.dot_compress_weight = nn.Parameter(torch.empty(2, input_dim // 2))
        nn.init.xavier_normal_(self.dot_compress_weight)
        
        self.dot_compress_bias = nn.Parameter(torch.zeros(input_dim // 2))

        self.dims = [input_dim] + hidden_dims + [1]
        self.output_layer = MLP(self.dims, apply_layernorm=True, elemwise_affine=True)
    
    def forward(self, set_embeddings, item_embeddings):
        all_embeddings = torch.stack([set_embeddings, item_embeddings], dim=1)
        combined_representation = torch.matmul(all_embeddings, torch.matmul(all_embeddings.transpose(1, 2), self.dot_compress_weight) + self.dot_compress_bias).flatten(1)
        output = self.output_layer(combined_representation)
        return output
    
class DotProduct(nn.Module):
    def __init__(self):
        super(DotProduct, self).__init__()
    
    def forward(self, set_embeddings, item_embeddings):
        return torch.sum(set_embeddings * item_embeddings, dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.W_q = nn.Linear(embed_dim, num_heads * embed_dim)
        self.W_k = nn.Linear(embed_dim, num_heads * embed_dim)
        self.W_v = nn.Linear(embed_dim, num_heads * embed_dim)
        
        self.num_heads =  num_heads
        self.embed_dim = embed_dim

    def forward(self, src):
        b_sz = src.shape[0]

        Q = self.W_q(src) # [B, S, H * D]
        K = self.W_k(src) # [B, S, H * D]
        V = self.W_v(src) # [B, S, H * D]

        Q = Q.view(b_sz, Q.shape[1], self.num_heads, self.embed_dim) # [B, S, H, D]
        K = K.view(b_sz, K.shape[1], self.num_heads, self.embed_dim) # [B, S, H, D]
        V = V.view(b_sz, V.shape[1], self.num_heads, self.embed_dim) # [B, S, H, D]

        Q = Q.permute(0, 2, 1, 3) # [B, H, S, D]
        K = K.permute(0, 2, 3, 1) # [B, H, D, S]
        V = V.permute(0, 2, 1, 3) # [B, H, S, D]

        scores = torch.matmul(Q, K) / (self.embed_dim ** 0.5) # [B, H, S, S]
        scores = torch.nn.functional.softmax(scores, dim=-1) # [B, H, S, S]

        out = torch.matmul(scores, V).transpose(0, 1).flatten(2) # [H, B, S * D]
        return out

class UnusableFeaturePredictor(nn.Module):
    def __init__(self, feat_embed_dim, num_usable_feats, num_cats_per_unusable_feat) -> None:
        super().__init__()
        self.num_cats_per_unusable_feat = num_cats_per_unusable_feat
        self.encoder = MultiHeadAttention(feat_embed_dim, len(num_cats_per_unusable_feat))
        self.ufp_output_mlps = nn.ModuleDict({feat_name : MLP([num_usable_feats * feat_embed_dim, num_cats], act="relu") for feat_name, num_cats in num_cats_per_unusable_feat.items()})
    
    def forward(self, feat_seq):
        attn_out = self.encoder(feat_seq)
        unknown_feat_predictions = {feat_name : self.ufp_output_mlps[feat_name](attn_out[i]) for i, feat_name in enumerate(self.num_cats_per_unusable_feat)}
        return unknown_feat_predictions




