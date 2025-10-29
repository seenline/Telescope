"""
HtapFormer Encoder Layer

This module implements the encoder layer for HtapFormer, which extends QueryFormer's
EncoderLayer by replacing the standard multi-head attention with HTAP-Bias Attention.

Architecture:
1. Layer Normalization
2. HTAP-Bias Multi-Head Attention (with tree bias + HTAP semantic bias)
3. Residual Connection + Dropout
4. Layer Normalization
5. Feed-Forward Network (FFN)
6. Residual Connection + Dropout

Corresponds to Section 4.2 in HtapFormer paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.htap_bias_attention import HTAPBiasAttention


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network (same as QueryFormer).
    
    FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
    """
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
    """
    HtapFormer Encoder Layer with HTAP-Bias Attention.
    
    This layer replaces QueryFormer's standard multi-head attention with
    HTAP-aware attention that considers:
    - Tree structural bias (hierarchical relationships in query plan)
    - Storage mode bias (row-store vs column-store performance)
    - Operator type bias (transactional vs analytical operations)
    """
    
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, 
                 head_size, storage_feature_dim=8, operator_feature_dim=8, 
                 htap_lambda=0.1):
        """
        Args:
            hidden_size: dimension of hidden representations
            ffn_size: dimension of feed-forward network
            dropout_rate: dropout rate for residual connections
            attention_dropout_rate: dropout rate for attention weights
            head_size: number of attention heads
            storage_feature_dim: dimension of storage mode features
            operator_feature_dim: dimension of operator type features
            htap_lambda: scaling coefficient λ for HTAP bias
        """
        super(HtapEncoderLayer, self).__init__()
        
        # Layer normalization before attention
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        
        # HTAP-Bias Multi-Head Attention
        self.self_attention = HTAPBiasAttention(
            hidden_size=hidden_size,
            attention_dropout_rate=attention_dropout_rate,
            head_size=head_size,
            storage_feature_dim=storage_feature_dim,
            operator_feature_dim=operator_feature_dim,
            htap_lambda=htap_lambda
        )
        
        # Dropout after attention
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        
        # Layer normalization before FFN
        self.ffn_norm = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        
        # Dropout after FFN
        self.ffn_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, tree_attn_bias=None, storage_features=None, operator_features=None):
        """
        Forward pass of HtapFormer encoder layer.
        
        Args:
            x: input node representations [batch, n_nodes, hidden_size]
            tree_attn_bias: tree structural bias [batch, head_size, n_nodes, n_nodes]
            storage_features: storage mode features [batch, n_nodes, storage_dim]
            operator_features: operator type features [batch, n_nodes, operator_dim]
            
        Returns:
            output: transformed representations [batch, n_nodes, hidden_size]
        """
        # Self-attention block with HTAP bias
        # 1. Layer normalization
        y = self.self_attention_norm(x)
        
        # 2. HTAP-Bias attention
        y = self.self_attention(
            q=y, k=y, v=y,
            tree_attn_bias=tree_attn_bias,
            storage_features=storage_features,
            operator_features=operator_features
        )
        
        # 3. Dropout + residual connection
        y = self.self_attention_dropout(y)
        x = x + y
        
        # Feed-forward block
        # 4. Layer normalization
        y = self.ffn_norm(x)
        
        # 5. FFN
        y = self.ffn(y)
        
        # 6. Dropout + residual connection
        y = self.ffn_dropout(y)
        x = x + y
        
        return x


class HtapEncoderStack(nn.Module):
    """
    Stack of HtapFormer encoder layers.
    
    This module stacks multiple HtapEncoderLayers to form the complete
    encoder for deep hierarchical feature learning.
    """
    
    def __init__(self, n_layers, hidden_size, ffn_size, dropout_rate, 
                 attention_dropout_rate, head_size, storage_feature_dim=8, 
                 operator_feature_dim=8, htap_lambda=0.1):
        """
        Args:
            n_layers: number of encoder layers
            hidden_size: dimension of hidden representations
            ffn_size: dimension of feed-forward network
            dropout_rate: dropout rate for residual connections
            attention_dropout_rate: dropout rate for attention weights
            head_size: number of attention heads
            storage_feature_dim: dimension of storage mode features
            operator_feature_dim: dimension of operator type features
            htap_lambda: scaling coefficient λ for HTAP bias
        """
        super(HtapEncoderStack, self).__init__()
        
        # Create stack of encoder layers
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
        
        # Final layer normalization
        self.final_ln = nn.LayerNorm(hidden_size)
        
    def forward(self, x, tree_attn_bias=None, storage_features=None, operator_features=None):
        """
        Forward pass through all encoder layers.
        
        Args:
            x: input node representations [batch, n_nodes, hidden_size]
            tree_attn_bias: tree structural bias [batch, head_size, n_nodes, n_nodes]
            storage_features: storage mode features [batch, n_nodes, storage_dim]
            operator_features: operator type features [batch, n_nodes, operator_dim]
            
        Returns:
            output: final encoded representations [batch, n_nodes, hidden_size]
        """
        # Pass through each encoder layer
        output = x
        for enc_layer in self.layers:
            output = enc_layer(
                x=output,
                tree_attn_bias=tree_attn_bias,
                storage_features=storage_features,
                operator_features=operator_features
            )
        
        # Final layer normalization
        output = self.final_ln(output)
        
        return output

