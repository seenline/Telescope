"""
HTAP Utility Functions for HtapFormer

This module provides utility functions for computing HTAP-specific features:
- Storage Bias computation: models write-intensive operation costs on different storage modes
- Operator Bias computation: combines query-level and node-level operator semantics
- General utilities: normalization, seeding, data preprocessing

Based on HtapFormer paper Section 4.2: HTAP-Bias Attention

NOTE: This file also includes utilities from QueryFormer's util.py for self-containment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# QueryFormer Utilities (from util.py)
# ============================================================================

class Normalizer():
    """
    Normalizer for cost and cardinality labels.
    Reused from QueryFormer.
    """
    def __init__(self, mini=None, maxi=None):
        self.mini = mini
        self.maxi = maxi
        
    def normalize_labels(self, labels, reset_min_max=False):
        """Normalize labels to [0, 1] range using log transformation."""
        # Added 0.001 for numerical stability
        labels = np.array([np.log(float(l) + 0.001) for l in labels])
        if self.mini is None or reset_min_max:
            self.mini = labels.min()
            print("min log(label): {}".format(self.mini))
        if self.maxi is None or reset_min_max:
            self.maxi = labels.max()
            print("max log(label): {}".format(self.maxi))
        labels_norm = (labels - self.mini) / (self.maxi - self.mini)
        # Threshold labels
        labels_norm = np.minimum(labels_norm, 1)
        labels_norm = np.maximum(labels_norm, 0.001)
        return labels_norm

    def unnormalize_labels(self, labels_norm):
        """Convert normalized labels back to original scale."""
        labels_norm = np.array(labels_norm, dtype=np.float32)
        labels = (labels_norm * (self.maxi - self.mini)) + self.mini
        return np.array(np.exp(labels) - 0.001)


def seed_everything(seed=0):
    """
    Set random seeds for reproducibility.
    Reused from QueryFormer.
    """
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False


def normalize_data(val, column_name, column_min_max_vals):
    """
    Normalize a value based on column min/max statistics.
    Reused from QueryFormer.
    """
    min_val = column_min_max_vals[column_name][0]
    max_val = column_min_max_vals[column_name][1]
    val = float(val)
    val_norm = 0.0
    if max_val > min_val:
        val_norm = (val - min_val) / (max_val - min_val)
    return np.array(val_norm, dtype=np.float32)


# ============================================================================
# HTAP-Specific Components
# ============================================================================

class StorageBiasComputer(nn.Module):
    """
    Computes storage mode bias v_j^s for each node.
    
    Storage bias models the performance impact of write-intensive operations
    on column-store nodes. For row-store or read-only operations, bias is zero.
    
    Formula (from paper):
    v_j^s = α_i * cnt_i + α_u * cnt_u + α_d * cnt_d
    
    where:
    - cnt_i, cnt_u, cnt_d: estimated execution frequencies of INSERT, UPDATE, DELETE
    - α_i, α_u, α_d: write-sensitivity coefficients (learnable)
    """
    
    def __init__(self, feature_dim=8):
        """
        Args:
            feature_dim: output dimension of storage features
        """
        super(StorageBiasComputer, self).__init__()
        
        # Write-sensitivity coefficients (learnable parameters)
        # These are initialized based on offline profiling and refined during training
        # Higher values indicate stronger cost sensitivity for the operation type
        self.alpha_insert = nn.Parameter(torch.tensor([]))  # INSERT sensitivity
        self.alpha_update = nn.Parameter(torch.tensor([]))  # UPDATE sensitivity
        self.alpha_delete = nn.Parameter(torch.tensor([]))  # DELETE sensitivity
        
        # Storage type embedding (0: row-store, 1: column-store, 2: hybrid)
        self.storage_embed = nn.Embedding(3, feature_dim)
        
        # MLP to project storage bias scalar to feature vector
        self.storage_mlp = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, storage_type, insert_cnt, update_cnt, delete_cnt):
        """
        Compute storage mode features for each node.
        
        Args:
            storage_type: storage mode for each node [batch, n_nodes]
                         0: row-store, 1: column-store, 2: hybrid
            insert_cnt: estimated INSERT frequency [batch, n_nodes]
            update_cnt: estimated UPDATE frequency [batch, n_nodes]
            delete_cnt: estimated DELETE frequency [batch, n_nodes]
            
        Returns:
            storage_features: [batch, n_nodes, feature_dim]
        """
        batch_size = storage_type.size(0)
        n_nodes = storage_type.size(1)
        
        # Compute write-intensity score
        # v_j^s = α_i * cnt_i + α_u * cnt_u + α_d * cnt_d
        write_intensity = (
            self.alpha_insert * insert_cnt + 
            self.alpha_update * update_cnt + 
            self.alpha_delete * delete_cnt
        )  # [batch, n_nodes]
        
        # For column-store nodes, apply write amplification penalty
        # For row-store nodes, set bias to zero (no write amplification)
        is_column_store = (storage_type == 1).float()  # 1 if column-store, 0 otherwise
        storage_bias = write_intensity * is_column_store  # [batch, n_nodes]
        
        # Project scalar bias to feature vector
        storage_bias_features = self.storage_mlp(storage_bias.unsqueeze(-1))  # [batch, n_nodes, feature_dim]
        
        # Add storage type embedding
        storage_type_embed = self.storage_embed(storage_type.long())  # [batch, n_nodes, feature_dim]
        
        # Combine bias and embedding
        storage_features = storage_bias_features + storage_type_embed
        
        return storage_features


class OperatorBiasComputer(nn.Module):
    """
    Computes operator type bias v_j^o for each node.
    
    Combines query-level context (INSERT, UPDATE, DELETE, SELECT) with
    node-level analytical operators (Scan, Join, Aggregate, GroupBy).
    
    Formula (from paper):
    v_j^o = w_{q_j} × w_{a_j}
    
    where:
    - q_j: query-level type (transactional context)
    - a_j: node-level analytical operator
    - w: learnable weights initialized by operation frequencies
    """
    
    def __init__(self, feature_dim=8):
        """
        Args:
            feature_dim: output dimension of operator features
        """
        super(OperatorBiasComputer, self).__init__()
        
        # Query-level type weights (transactional context)
        # Types: INSERT, UPDATE, DELETE, SELECT
        self.query_type_embed = nn.Embedding(4, feature_dim)
        
        # Node-level analytical operator weights
        # Types: Scan, Join, Aggregate, GroupBy, Sort, Filter, etc.
        # We use a more comprehensive set to cover common operators
        self.node_operator_embed = nn.Embedding(20, feature_dim)
        
        # Interaction MLP to combine query type and node operator
        self.operator_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, query_type, node_operator):
        """
        Compute operator type features for each node.
        
        Args:
            query_type: query-level type [batch, n_nodes]
                       0: INSERT, 1: UPDATE, 2: DELETE, 3: SELECT
            node_operator: node-level operator [batch, n_nodes]
                          0: Scan, 1: Join, 2: Aggregate, 3: GroupBy, etc.
            
        Returns:
            operator_features: [batch, n_nodes, feature_dim]
        """
        # Get embeddings
        query_embed = self.query_type_embed(query_type.long())  # [batch, n_nodes, feature_dim]
        operator_embed = self.node_operator_embed(node_operator.long())  # [batch, n_nodes, feature_dim]
        
        # Combine: v_j^o = w_{q_j} × w_{a_j}
        # We use element-wise multiplication followed by MLP for richer interaction
        combined = torch.cat([query_embed, operator_embed * query_embed], dim=-1)
        operator_features = self.operator_mlp(combined)
        
        return operator_features


class HTAPFeatureExtractor(nn.Module):
    """
    High-level feature extractor that combines storage and operator bias computation.
    
    This module provides a unified interface for extracting HTAP-specific features
    from query plan nodes, which are then used in HTAP-Bias Attention.
    """
    
    def __init__(self, storage_feature_dim=8, operator_feature_dim=8):
        """
        Args:
            storage_feature_dim: dimension of storage features
            operator_feature_dim: dimension of operator features
        """
        super(HTAPFeatureExtractor, self).__init__()
        
        self.storage_computer = StorageBiasComputer(storage_feature_dim)
        self.operator_computer = OperatorBiasComputer(operator_feature_dim)
        
    def forward(self, node_data):
        """
        Extract HTAP features from node data.
        
        Args:
            node_data: dictionary containing:
                - 'storage_type': [batch, n_nodes] storage mode
                - 'insert_cnt': [batch, n_nodes] INSERT frequency
                - 'update_cnt': [batch, n_nodes] UPDATE frequency
                - 'delete_cnt': [batch, n_nodes] DELETE frequency
                - 'query_type': [batch, n_nodes] query-level type
                - 'node_operator': [batch, n_nodes] node-level operator
                
        Returns:
            storage_features: [batch, n_nodes, storage_feature_dim]
            operator_features: [batch, n_nodes, operator_feature_dim]
        """
        # Compute storage bias features
        storage_features = self.storage_computer(
            storage_type=node_data['storage_type'],
            insert_cnt=node_data['insert_cnt'],
            update_cnt=node_data['update_cnt'],
            delete_cnt=node_data['delete_cnt']
        )
        
        # Compute operator bias features
        operator_features = self.operator_computer(
            query_type=node_data['query_type'],
            node_operator=node_data['node_operator']
        )
        
        return storage_features, operator_features


def compute_sensitivity_coefficients(cost_profile):
    """
    Compute write-sensitivity coefficients from offline profiling data.
    
    Formula (from paper):
    α_t = (C_t / C_ref) / Σ_{k∈{i,u,d}} (C_k / C_ref)
    
    where:
    - C_t: average cost of operation type t
    - C_ref: baseline cost of read-only operation (SELECT)
    
    Args:
        cost_profile: dictionary with keys 'insert', 'update', 'delete', 'select'
                     containing average costs for each operation type
                     
    Returns:
        alpha_dict: dictionary with normalized sensitivity coefficients
    """
    # Baseline cost (read-only operation)
    C_ref = cost_profile.get('select', 1.0)
    
    # Normalized costs
    C_insert = cost_profile.get('insert', 1.0) / C_ref
    C_update = cost_profile.get('update', 1.0) / C_ref
    C_delete = cost_profile.get('delete', 1.0) / C_ref
    
    # Sum for normalization
    total = C_insert + C_update + C_delete
    
    # Compute sensitivity coefficients
    alpha_dict = {
        'alpha_insert': C_insert / total,
        'alpha_update': C_update / total,
        'alpha_delete': C_delete / total
    }
    
    return alpha_dict


def estimate_operation_frequency(workload_history, window_size=1000):
    """
    Estimate operation frequencies from workload history.
    
    This is a simple frequency estimator. In practice, you might use
    a more sophisticated model (e.g., linear regression, time-series model).
    
    Args:
        workload_history: list of recent operations with their types
        window_size: number of recent operations to consider
        
    Returns:
        frequency_dict: estimated frequencies for INSERT, UPDATE, DELETE
    """
    recent_workload = workload_history[-window_size:] if len(workload_history) > window_size else workload_history
    
    total = len(recent_workload)
    if total == 0:
        return {'insert': 0.0, 'update': 0.0, 'delete': 0.0}
    
    insert_cnt = sum(1 for op in recent_workload if op['type'] == 'INSERT')
    update_cnt = sum(1 for op in recent_workload if op['type'] == 'UPDATE')
    delete_cnt = sum(1 for op in recent_workload if op['type'] == 'DELETE')
    
    return {
        'insert': insert_cnt / total,
        'update': update_cnt / total,
        'delete': delete_cnt / total
    }
