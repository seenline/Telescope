"""
HTAP-Bias Attention Module for HtapFormer

This module implements the HTAP-aware attention mechanism described in the HtapFormer paper.
Key components:
- Storage Bias: models storage-mode interactions (row-store vs column-store)
- Operator-Type Bias: captures operator-type interactions (INSERT, UPDATE, DELETE, SELECT, etc.)
- HTAP Interaction Functions: f_s and f_o (MLP-based learnable functions)

Attention formula:
A'_{ij} = (Q_i K_j^T) / sqrt(d) + b_tree(d_{ij}) + λ * b_HTAP(i, j)

where:
- b_tree: tree structural bias (from QueryFormer)
- b_HTAP: HTAP semantic bias = f_s(v_i^s, v_j^s) + f_o(v_i^o, v_j^o)
- λ: scaling coefficient balancing structural and semantic biases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HTAPInteractionMLP(nn.Module):
    """
    MLP-based interaction function for computing pairwise semantic bias.
    
    Takes two node features and computes their interaction score through:
    MLP([v_i || v_j || |v_i - v_j|])
    
    This captures both the individual features and their difference,
    enabling the model to learn symmetric or asymmetric interactions.
    """
    def __init__(self, feature_dim, hidden_dim=64, output_dim=1):
        """
        Args:
            feature_dim: dimension of input node features
            hidden_dim: hidden layer dimension
            output_dim: output dimension (typically 1 for scalar bias)
        """
        super(HTAPInteractionMLP, self).__init__()
        
        # Input is concatenation of [v_i, v_j, |v_i - v_j|]
        input_dim = feature_dim * 3
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, v_i, v_j):
        """
        Compute interaction score between node i and node j.
        
        Args:
            v_i: features of node i, shape [batch, n_nodes, feature_dim] or [batch, feature_dim]
            v_j: features of node j, same shape as v_i
            
        Returns:
            interaction score, shape matches input but with last dim = output_dim
        """
        # Compute absolute difference
        diff = torch.abs(v_i - v_j)
        
        # Concatenate [v_i, v_j, |v_i - v_j|]
        concat_features = torch.cat([v_i, v_j, diff], dim=-1)
        
        # Apply MLP
        return self.mlp(concat_features)


class HTAPBiasAttention(nn.Module):
    """
    HTAP-Bias Multi-Head Attention mechanism.
    
    Extends the standard multi-head attention with HTAP semantic biases:
    - Storage mode bias: models row-store vs column-store performance characteristics
    - Operator type bias: captures transactional vs analytical operation semantics
    
    The attention is computed as:
    A'_{ij} = (Q_i K_j^T) / sqrt(d) + b_tree + λ * b_HTAP
    
    where b_HTAP = f_s(v_i^s, v_j^s) + f_o(v_i^o, v_j^o)
    """
    
    def __init__(self, hidden_size, attention_dropout_rate, head_size, 
                 storage_feature_dim=8, operator_feature_dim=8,
                 htap_lambda=0.1):
        """
        Args:
            hidden_size: dimension of hidden representations
            attention_dropout_rate: dropout rate for attention weights
            head_size: number of attention heads
            storage_feature_dim: dimension of storage mode features
            operator_feature_dim: dimension of operator type features
            htap_lambda: scaling coefficient λ for HTAP bias term
        """
        super(HTAPBiasAttention, self).__init__()
        
        self.head_size = head_size
        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5
        self.htap_lambda = htap_lambda
        
        # Standard attention projections (Q, K, V)
        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(head_size * att_size, hidden_size)
        
        # HTAP Interaction Functions
        # f_s: Storage mode interaction function
        self.f_s = HTAPInteractionMLP(
            feature_dim=storage_feature_dim,
            hidden_dim=64,
            output_dim=head_size  # One bias value per head
        )
        
        # f_o: Operator type interaction function
        self.f_o = HTAPInteractionMLP(
            feature_dim=operator_feature_dim,
            hidden_dim=64,
            output_dim=head_size
        )
        
    def compute_htap_bias(self, storage_features, operator_features):
        """
        Compute HTAP semantic bias b_HTAP(i, j) for all pairs of nodes.
        
        b_HTAP(i, j) = f_s(v_i^s, v_j^s) + f_o(v_i^o, v_j^o)
        
        Args:
            storage_features: [batch, n_nodes, storage_feature_dim]
            operator_features: [batch, n_nodes, operator_feature_dim]
            
        Returns:
            htap_bias: [batch, head_size, n_nodes, n_nodes]
        """
        batch_size = storage_features.size(0)
        n_nodes = storage_features.size(1)
        
        # Expand features for pairwise computation
        # v_i: [batch, n_nodes, 1, feature_dim]
        # v_j: [batch, 1, n_nodes, feature_dim]
        storage_i = storage_features.unsqueeze(2)  # [batch, n_nodes, 1, dim]
        storage_j = storage_features.unsqueeze(1)  # [batch, 1, n_nodes, dim]
        
        operator_i = operator_features.unsqueeze(2)
        operator_j = operator_features.unsqueeze(1)
        
        # Expand to all pairs
        storage_i = storage_i.expand(batch_size, n_nodes, n_nodes, -1)
        storage_j = storage_j.expand(batch_size, n_nodes, n_nodes, -1)
        operator_i = operator_i.expand(batch_size, n_nodes, n_nodes, -1)
        operator_j = operator_j.expand(batch_size, n_nodes, n_nodes, -1)
        
        # Compute storage bias: f_s(v_i^s, v_j^s)
        # Output: [batch, n_nodes, n_nodes, head_size]
        storage_bias = self.f_s(storage_i, storage_j)
        
        # Compute operator bias: f_o(v_i^o, v_j^o)
        operator_bias = self.f_o(operator_i, operator_j)
        
        # Combine biases: b_HTAP = f_s + f_o
        # [batch, n_nodes, n_nodes, head_size]
        htap_bias = storage_bias + operator_bias
        
        # Transpose to [batch, head_size, n_nodes, n_nodes] for attention
        htap_bias = htap_bias.permute(0, 3, 1, 2)
        
        return htap_bias
    
    def forward(self, q, k, v, tree_attn_bias=None, 
                storage_features=None, operator_features=None):
        """
        Forward pass of HTAP-Bias Attention.
        
        Args:
            q, k, v: query, key, value tensors [batch, n_nodes, hidden_size]
            tree_attn_bias: tree structural bias from QueryFormer [batch, head_size, n_nodes, n_nodes]
            storage_features: storage mode features [batch, n_nodes, storage_dim]
            operator_features: operator type features [batch, n_nodes, operator_dim]
            
        Returns:
            attention output: [batch, n_nodes, hidden_size]
        """
        orig_q_size = q.size()
        
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        
        # Linear projections and reshape to multi-head format
        # [batch, n_nodes, head_size, d_k]
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, head_size, n_nodes, d_k]
        v = v.transpose(1, 2)  # [batch, head_size, n_nodes, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [batch, head_size, d_k, n_nodes]
        
        # Scaled dot-product attention: QK^T / sqrt(d_k)
        q = q * self.scale
        attn_scores = torch.matmul(q, k)  # [batch, head_size, n_nodes, n_nodes]
        
        # Add tree structural bias (from QueryFormer)
        if tree_attn_bias is not None:
            attn_scores = attn_scores + tree_attn_bias
        
        # Compute and add HTAP semantic bias
        if storage_features is not None and operator_features is not None:
            htap_bias = self.compute_htap_bias(storage_features, operator_features)
            # Add with scaling coefficient λ
            attn_scores = attn_scores + self.htap_lambda * htap_bias
        
        # Apply softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.att_dropout(attn_weights)
        
        # Apply attention to values
        x = attn_weights.matmul(v)  # [batch, head_size, n_nodes, d_v]
        
        # Reshape back to [batch, n_nodes, hidden_size]
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.head_size * d_v)
        
        # Final output projection
        x = self.output_layer(x)
        
        assert x.size() == orig_q_size
        return x

