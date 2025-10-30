 

import torch
import torch.nn as nn
import torch.nn.functional as F


class HTAPInteractionMLP(nn.Module):
    
    def __init__(self, feature_dim, hidden_dim=64, output_dim=1):
        
        super(HTAPInteractionMLP, self).__init__()
        
        input_dim = feature_dim * 3
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, v_i, v_j):
        diff = torch.abs(v_i - v_j)
        concat_features = torch.cat([v_i, v_j, diff], dim=-1)
        return self.mlp(concat_features)


class HTAPBiasAttention(nn.Module):
    
    
    def __init__(self, hidden_size, attention_dropout_rate, head_size, 
                 storage_feature_dim=8, operator_feature_dim=8,
                 htap_lambda=0.1):
        super(HTAPBiasAttention, self).__init__()
        
        self.head_size = head_size
        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5
        self.htap_lambda = htap_lambda
        
        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(head_size * att_size, hidden_size)
        
        self.f_s = HTAPInteractionMLP(
            feature_dim=storage_feature_dim,
            hidden_dim=64,
            output_dim=head_size
        )
        
        self.f_o = HTAPInteractionMLP(
            feature_dim=operator_feature_dim,
            hidden_dim=64,
            output_dim=head_size
        )
        
    def compute_htap_bias(self, storage_features, operator_features):
        batch_size = storage_features.size(0)
        n_nodes = storage_features.size(1)
        
        storage_i = storage_features.unsqueeze(2)
        storage_j = storage_features.unsqueeze(1)
        
        operator_i = operator_features.unsqueeze(2)
        operator_j = operator_features.unsqueeze(1)
        
        storage_i = storage_i.expand(batch_size, n_nodes, n_nodes, -1)
        storage_j = storage_j.expand(batch_size, n_nodes, n_nodes, -1)
        operator_i = operator_i.expand(batch_size, n_nodes, n_nodes, -1)
        operator_j = operator_j.expand(batch_size, n_nodes, n_nodes, -1)
        
        storage_bias = self.f_s(storage_i, storage_j)
        
        operator_bias = self.f_o(operator_i, operator_j)
        
        htap_bias = storage_bias + operator_bias
        
        htap_bias = htap_bias.permute(0, 3, 1, 2)
        
        return htap_bias
    
    def forward(self, q, k, v, tree_attn_bias=None, 
                storage_features=None, operator_features=None):
        orig_q_size = q.size()
        
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)
        
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)
        
        q = q * self.scale
        attn_scores = torch.matmul(q, k)
        
        if tree_attn_bias is not None:
            attn_scores = attn_scores + tree_attn_bias
        
        if storage_features is not None and operator_features is not None:
            htap_bias = self.compute_htap_bias(storage_features, operator_features)
            attn_scores = attn_scores + self.htap_lambda * htap_bias
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.att_dropout(attn_weights)
        
        x = attn_weights.matmul(v)
        
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.head_size * d_v)
        
        x = self.output_layer(x)
        
        assert x.size() == orig_q_size
        return x

