 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalizer():
    def __init__(self, mini=None, maxi=None):
        self.mini = mini
        self.maxi = maxi
        
    def normalize_labels(self, labels, reset_min_max=False):
        labels = np.array([np.log(float(l) + 0.001) for l in labels])
        if self.mini is None or reset_min_max:
            self.mini = labels.min()
            print("min log(label): {}".format(self.mini))
        if self.maxi is None or reset_min_max:
            self.maxi = labels.max()
            print("max log(label): {}".format(self.maxi))
        labels_norm = (labels - self.mini) / (self.maxi - self.mini)
        labels_norm = np.minimum(labels_norm, 1)
        labels_norm = np.maximum(labels_norm, 0.001)
        return labels_norm

    def unnormalize_labels(self, labels_norm):
        labels_norm = np.array(labels_norm, dtype=np.float32)
        labels = (labels_norm * (self.maxi - self.mini)) + self.mini
        return np.array(np.exp(labels) - 0.001)


def seed_everything(seed=0):
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False


def normalize_data(val, column_name, column_min_max_vals):
    min_val = column_min_max_vals[column_name][0]
    max_val = column_min_max_vals[column_name][1]
    val = float(val)
    val_norm = 0.0
    if max_val > min_val:
        val_norm = (val - min_val) / (max_val - min_val)
    return np.array(val_norm, dtype=np.float32)



class StorageBiasComputer(nn.Module):
    
    
    def __init__(self, feature_dim=8):
        super(StorageBiasComputer, self).__init__()
        
        self.alpha_insert = nn.Parameter(torch.tensor([]))
        self.alpha_update = nn.Parameter(torch.tensor([]))
        self.alpha_delete = nn.Parameter(torch.tensor([]))
        
        self.storage_embed = nn.Embedding(3, feature_dim)
        
        self.storage_mlp = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, storage_type, insert_cnt, update_cnt, delete_cnt):
        batch_size = storage_type.size(0)
        n_nodes = storage_type.size(1)
        
        write_intensity = (
            self.alpha_insert * insert_cnt + 
            self.alpha_update * update_cnt + 
            self.alpha_delete * delete_cnt
        )
        
        is_column_store = (storage_type == 1).float()
        storage_bias = write_intensity * is_column_store
        
        storage_bias_features = self.storage_mlp(storage_bias.unsqueeze(-1))
        
        storage_type_embed = self.storage_embed(storage_type.long())
        
        storage_features = storage_bias_features + storage_type_embed
        
        return storage_features


class OperatorBiasComputer(nn.Module):
    
    def __init__(self, feature_dim=8):
        super(OperatorBiasComputer, self).__init__()
        
        self.query_type_embed = nn.Embedding(4, feature_dim)
        
        self.node_operator_embed = nn.Embedding(20, feature_dim)
        
        self.operator_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, query_type, node_operator):
        query_embed = self.query_type_embed(query_type.long())
        operator_embed = self.node_operator_embed(node_operator.long())
        
        combined = torch.cat([query_embed, operator_embed * query_embed], dim=-1)
        operator_features = self.operator_mlp(combined)
        
        return operator_features


class HTAPFeatureExtractor(nn.Module):
    
    def __init__(self, storage_feature_dim=8, operator_feature_dim=8):
        super(HTAPFeatureExtractor, self).__init__()
        
        self.storage_computer = StorageBiasComputer(storage_feature_dim)
        self.operator_computer = OperatorBiasComputer(operator_feature_dim)
        
    def forward(self, node_data):
        storage_features = self.storage_computer(
            storage_type=node_data['storage_type'],
            insert_cnt=node_data['insert_cnt'],
            update_cnt=node_data['update_cnt'],
            delete_cnt=node_data['delete_cnt']
        )
        
        operator_features = self.operator_computer(
            query_type=node_data['query_type'],
            node_operator=node_data['node_operator']
        )
        
        return storage_features, operator_features


def compute_sensitivity_coefficients(cost_profile):
    C_ref = cost_profile.get('select', 1.0)
    C_insert = cost_profile.get('insert', 1.0) / C_ref
    C_update = cost_profile.get('update', 1.0) / C_ref
    C_delete = cost_profile.get('delete', 1.0) / C_ref
    total = C_insert + C_update + C_delete
    alpha_dict = {
        'alpha_insert': C_insert / total,
        'alpha_update': C_update / total,
        'alpha_delete': C_delete / total
    }
    
    return alpha_dict


def estimate_operation_frequency(workload_history, window_size=1000):
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
