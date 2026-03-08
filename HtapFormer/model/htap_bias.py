"""
HTAP-Bias Attention Module
A'_{ij} = (Q_i K_j^T) / √d + b_tree(d_ij) + λ · b_HTAP(i,j)
b_HTAP(i,j) = f_s(v_i^s, v_j^s) + f_o(v_i^o, v_j^o)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HTAPBiasModule(nn.Module):
  
    def __init__(self, 
                 alpha_i=1.0, alpha_u=1.0, alpha_d=1.0,  
                 hidden_dim=32,  
                 lambda_scale=1.0,  
                 learnable_weights=True,  
                 query_weight_init=None,  
                 operator_weight_init=None):  
        super(HTAPBiasModule, self).__init__()
        
        self.lambda_scale = lambda_scale
        self.learnable_weights = learnable_weights
        
        self.register_buffer('alpha_i', torch.tensor(alpha_i, dtype=torch.float32))
        self.register_buffer('alpha_u', torch.tensor(alpha_u, dtype=torch.float32))
        self.register_buffer('alpha_d', torch.tensor(alpha_d, dtype=torch.float32))
        
        query_init_tensor = self._prepare_weight_init(query_weight_init, 4, 'query weights')
        operator_init_tensor = self._prepare_weight_init(operator_weight_init, 5, 'operator weights')
        
        # w_q: INSERT, UPDATE, DELETE, SELECT
        if learnable_weights:
            self.query_type_weights = nn.Parameter(query_init_tensor)
        else:
            self.register_buffer('query_type_weights', query_init_tensor)
        
        # w_a: Scan, Join, Aggregate, GroupBy, Other
        if learnable_weights:
            self.operator_weights = nn.Parameter(operator_init_tensor)
        else:
            self.register_buffer('operator_weights', operator_init_tensor)
        
        # f_s: MLP_s([v_i^s || v_j^s || |v_i^s - v_j^s|])
        self.mlp_s = nn.Sequential(
            nn.Linear(3, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  
        )
        
        # f_o: MLP_o([v_i^o || v_j^o || |v_i^o - v_j^o|])
        self.mlp_o = nn.Sequential(
            nn.Linear(3, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  
        )
        
        self.query_type_map = {'INSERT': 0, 'UPDATE': 1, 'DELETE': 2, 'SELECT': 3}
        self.operator_map = {'Scan': 0, 'Join': 1, 'Aggregate': 2, 'GroupBy': 3, 'Other': 4}
    
    @staticmethod
    def _prepare_weight_init(values, expected_len, name):
    
        if values is None:
            tensor = torch.ones(expected_len, dtype=torch.float32)
        elif isinstance(values, torch.Tensor):
            tensor = values.detach().clone().float()
        else:
            tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.numel() != expected_len:
            raise ValueError(f"{name} must have length {expected_len}, got {tensor.numel()}")
        return tensor
    
    def compute_storage_bias_value(self, storage_mode, write_counts):
        device = self.alpha_i.device
        
        if storage_mode == 'row-store' or storage_mode == 'NA':
            return torch.zeros(1, dtype=torch.float32, device=device).squeeze(0)
        
        cnt_i = write_counts.get('insert', 0)
        cnt_u = write_counts.get('update', 0)
        cnt_d = write_counts.get('delete', 0)
        
        if isinstance(cnt_i, (int, float)):
            cnt_i = torch.tensor(float(cnt_i), dtype=torch.float32, device=device)
        if isinstance(cnt_u, (int, float)):
            cnt_u = torch.tensor(float(cnt_u), dtype=torch.float32, device=device)
        if isinstance(cnt_d, (int, float)):
            cnt_d = torch.tensor(float(cnt_d), dtype=torch.float32, device=device)
        
        v_j_s = self.alpha_i * cnt_i + self.alpha_u * cnt_u + self.alpha_d * cnt_d
        return v_j_s
    
    def compute_operator_bias_value(self, query_type, node_operator):
    
        q_idx = self.query_type_map.get(query_type, 3)  
        a_idx = self.operator_map.get(node_operator, 4)  
        
        w_q = self.query_type_weights[q_idx]
        w_a = self.operator_weights[a_idx]
        
        #  v_j^o = w_q × w_a
        return (w_q * w_a).reshape(()).to(self.alpha_i.device)
    
    def _compute_interactions(self, values, mlp):
        n = values.size(0)
        if n == 0:
            return torch.zeros(0, 0, device=values.device)
        vi = values.view(n, 1)
        vj = values.view(1, n)
        concat = torch.stack(
            [vi.expand(n, n), vj.expand(n, n), torch.abs(vi - vj)],
            dim=-1
        )  # [n, n, 3]
        out = mlp(concat.view(-1, 3)).view(n, n, -1).squeeze(-1)
        return out
    
    def forward(self, htap_info_list):
        batch_size = len(htap_info_list)
        device = self.alpha_i.device
        valid_infos = [info for info in htap_info_list if info is not None]
        if len(valid_infos) == 0:
            return torch.zeros(batch_size, 1, 1, dtype=torch.float32, device=device)
        
        max_nodes = max(len(info.get('storage_modes', [])) for info in valid_infos)
        if max_nodes == 0:
            max_nodes = 1
        
        htap_bias = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.float32, device=device)
        
        for batch_idx, htap_info in enumerate(htap_info_list):
            if htap_info is None:
                continue
            
            storage_modes = htap_info.get('storage_modes', [])
            query_types = htap_info.get('query_types', [])
            node_operators = htap_info.get('node_operators', [])
            write_counts_list = htap_info.get('write_counts', [])
            
            n_nodes = len(storage_modes)
            if n_nodes == 0:
                continue
            
            storage_vals = []
            operator_vals = []
            for idx in range(n_nodes):
                storage_mode = storage_modes[idx] if idx < len(storage_modes) else 'NA'
                query_type = query_types[idx] if idx < len(query_types) else 'SELECT'
                node_operator = node_operators[idx] if idx < len(node_operators) else 'Other'
                write_counts = write_counts_list[idx] if idx < len(write_counts_list) else {'insert': 0, 'update': 0, 'delete': 0}
                
                storage_vals.append(self.compute_storage_bias_value(storage_mode, write_counts))
                operator_vals.append(self.compute_operator_bias_value(query_type, node_operator))
            
            storage_tensor = torch.stack(storage_vals)  # [n_nodes]
            operator_tensor = torch.stack(operator_vals)  # [n_nodes]
            
            f_s_matrix = self._compute_interactions(storage_tensor, self.mlp_s)
            f_o_matrix = self._compute_interactions(operator_tensor, self.mlp_o)
            
            htap_bias[batch_idx, :n_nodes, :n_nodes] = f_s_matrix + f_o_matrix
        
        return htap_bias

