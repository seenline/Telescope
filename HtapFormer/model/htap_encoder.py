 

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.htap_bias_attention import HTAPBiasAttention


class FeedForwardNetwork(nn.Module):
    
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class HtapEncoderLayer(nn.Module):
    
    
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, 
                 head_size, storage_feature_dim=8, operator_feature_dim=8, 
                 htap_lambda=0.1):
        super(HtapEncoderLayer, self).__init__()
        
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        
        self.self_attention = HTAPBiasAttention(
            hidden_size=hidden_size,
            attention_dropout_rate=attention_dropout_rate,
            head_size=head_size,
            storage_feature_dim=storage_feature_dim,
            operator_feature_dim=operator_feature_dim,
            htap_lambda=htap_lambda
        )
        
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        
        self.ffn_norm = nn.LayerNorm(hidden_size)
        
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        
        self.ffn_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, tree_attn_bias=None, storage_features=None, operator_features=None):
        y = self.self_attention_norm(x)
        
        y = self.self_attention(
            q=y, k=y, v=y,
            tree_attn_bias=tree_attn_bias,
            storage_features=storage_features,
            operator_features=operator_features
        )
        
        y = self.self_attention_dropout(y)
        x = x + y
        
        y = self.ffn_norm(x)
        
        y = self.ffn(y)
        
        y = self.ffn_dropout(y)
        x = x + y
        
        return x


class HtapEncoderStack(nn.Module):
    
    
    def __init__(self, n_layers, hidden_size, ffn_size, dropout_rate, 
                 attention_dropout_rate, head_size, storage_feature_dim=8, 
                 operator_feature_dim=8, htap_lambda=0.1):
        super(HtapEncoderStack, self).__init__()
        
        encoders = [
            HtapEncoderLayer(
                hidden_size=hidden_size,
                ffn_size=ffn_size,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                head_size=head_size,
                storage_feature_dim=storage_feature_dim,
                operator_feature_dim=operator_feature_dim,
                htap_lambda=htap_lambda
            )
            for _ in range(n_layers)
        ]
        
        self.layers = nn.ModuleList(encoders)
        
        self.final_ln = nn.LayerNorm(hidden_size)
        
    def forward(self, x, tree_attn_bias=None, storage_features=None, operator_features=None):
        output = x
        for enc_layer in self.layers:
            output = enc_layer(
                x=output,
                tree_attn_bias=tree_attn_bias,
                storage_features=storage_features,
                operator_features=operator_features
            )
        
        output = self.final_ln(output)
        
        return output

