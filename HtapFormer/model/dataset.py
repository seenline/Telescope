import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
import sys, os
from collections import deque
from .database_util import formatFilter, formatJoin, TreeNode, filterDict2Hist, map_node_type_to_operator
from .database_util import *

class PlanTreeDataset(Dataset):
    def __init__(self, json_df : pd.DataFrame, train : pd.DataFrame, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample):

        self.table_sample = table_sample
        self.encoding = encoding
        self.hist_file = hist_file
        
        self.length = len(json_df)
        # train = train.loc[json_df['id']]
        
        nodes = [json.loads(plan)['Plan'] for plan in json_df['json']]
        self.cards = [node['Actual Rows'] for node in nodes]
        self.costs = [json.loads(plan)['Execution Time'] for plan in json_df['json']]
        
        self.card_labels = torch.from_numpy(card_norm.normalize_labels(self.cards))
        self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(self.costs))
        
        self.to_predict = to_predict
        if to_predict == 'cost':
            self.gts = self.costs
            self.labels = self.cost_labels
        elif to_predict == 'card':
            self.gts = self.cards
            self.labels = self.card_labels
        elif to_predict == 'both': ## try not to use, just in case
            self.gts = self.costs
            self.labels = self.cost_labels
        else:
            raise Exception('Unknown to_predict type')
            
        idxs = list(json_df['id'])
        

        if 'query_type' in json_df.columns and 'storage_mode' in json_df.columns:

            json_df_indexed = json_df.set_index('id')
            self.htap_assignments = {
                idx: {
                    'query_type': json_df_indexed.loc[idx, 'query_type'],
                    'storage_mode': json_df_indexed.loc[idx, 'storage_mode']
                }
                for idx in idxs
            }

            cumulative_counts = {'insert': 0, 'update': 0, 'delete': 0}
            self.per_query_cumulative_counts = {}
            for _, row in json_df.iterrows():
                idx = row['id']
                qtype = row.get('query_type', 'SELECT')
                # Increment counters based on current query type
                if isinstance(qtype, str):
                    q_upper = qtype.upper()
                    if q_upper == 'INSERT':
                        cumulative_counts['insert'] += 1
                    elif q_upper == 'UPDATE':
                        cumulative_counts['update'] += 1
                    elif q_upper == 'DELETE':
                        cumulative_counts['delete'] += 1
                self.per_query_cumulative_counts[idx] = cumulative_counts.copy()
        else:
            print("Warning: query_type and storage_mode not found in CSV, using default values")
            self.htap_assignments = {
                idx: {
                    'query_type': 'SELECT',
                    'storage_mode': 'row-store'
                }
                for idx in idxs
            }
            self.per_query_cumulative_counts = {
                idx: {'insert': 0, 'update': 0, 'delete': 0}
                for idx in idxs
            }
    
        self.treeNodes = [] ## for mem collection
        self.collated_dicts = [self.js_node2dict(i,node) for i,node in zip(idxs, nodes)]

    def js_node2dict(self, idx, node):
        treeNode = self.traversePlan(node, idx, self.encoding)
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict)
        
        self.treeNodes.clear()
        del self.treeNodes[:]

        return collated_dict

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        return self.collated_dicts[idx], (self.cost_labels[idx], self.card_labels[idx])

    def old_getitem(self, idx):
        return self.dicts[idx], (self.cost_labels[idx], self.card_labels[idx])
      
    ## pre-process first half of old collator
    def pre_collate(self, the_dict, max_node = 30, rel_pos_max = 20):

        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N+1,N+1], dtype=torch.float)
        
        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N,N], dtype=torch.bool)
            adj[edge_index[0,:], edge_index[1,:]] = True
            
            shortest_path_result = floyd_warshall_rewrite(adj.numpy())
        
        rel_pos = torch.from_numpy((shortest_path_result)).long()

        
        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')
        
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)
        
    
        htap_info = the_dict.get('htap_info', {})
        if htap_info:
         
            storage_modes = htap_info.get('storage_modes', [])
            query_types = htap_info.get('query_types', [])
            node_operators = htap_info.get('node_operators', [])
            write_counts = htap_info.get('write_counts', [])
            
        
            if len(storage_modes) > max_node:
                storage_modes = storage_modes[:max_node]
                query_types = query_types[:max_node]
                node_operators = node_operators[:max_node]
             
            elif len(storage_modes) < max_node:
                last_storage = storage_modes[-1] if storage_modes else 'NA'
                last_query = query_types[-1] if query_types else 'SELECT'
                last_operator = node_operators[-1] if node_operators else 'Other'
                last_write_counts = write_counts[-1] if write_counts else {'insert': 0, 'update': 0, 'delete': 0}
                
                storage_modes.extend([last_storage] * (max_node - len(storage_modes)))
                query_types.extend([last_query] * (max_node - len(query_types)))
                node_operators.extend([last_operator] * (max_node - len(node_operators)))
                write_counts.extend([last_write_counts.copy()] * (max_node - len(write_counts)))
            
            htap_info = {
                'storage_modes': storage_modes,
                'query_types': query_types,
                'node_operators': node_operators,
                'write_counts': write_counts
            }
        else:
            
            htap_info = {
                'storage_modes': ['NA'] * max_node,
                'query_types': ['SELECT'] * max_node,
                'node_operators': ['Other'] * max_node,
                'write_counts': [{'insert': 0, 'update': 0, 'delete': 0}] * max_node
            }
        
        return {
            'x' : x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights,
            'htap_info': htap_info  
        }


    def node2dict(self, treeNode):

        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))
        
        storage_modes = []
        query_types = []
        node_operators = []
        write_counts_list = []
        
        toVisit = deque()
        toVisit.append(treeNode)
        while toVisit:
            node = toVisit.popleft()
            storage_modes.append(node.storage_mode)
            query_types.append(node.query_type)
            node_operators.append(node.node_operator)
            write_counts_list.append(node.write_counts)
            for child in node.children:
                toVisit.append(child)

        return {
            'features' : torch.FloatTensor(np.array(features)),
            'heights' : torch.LongTensor(heights),
            'adjacency_list' : torch.LongTensor(np.array(adj_list)),
            'htap_info': {  
                'storage_modes': storage_modes,
                'query_types': query_types,
                'node_operators': node_operators,
                'write_counts': write_counts_list
            }
        }
    
    def topo_sort(self, root_node):
#        nodes = []
        adj_list = [] #from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0,root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
#            nodes.append(node)
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id,child))
                adj_list.append((idx,next_id))
                next_id += 1
        
        return adj_list, num_child, features
    
    def traversePlan(self, plan, idx, encoding): # bfs accumulate plan

        nodeType = plan['Node Type']
        typeId = encoding.encode_type(nodeType)
        card = None #plan['Actual Rows']
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)
        
        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)
        
        self.treeNodes.append(root)

        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx
        
        root.node_operator = map_node_type_to_operator(nodeType)
        
        if idx in self.htap_assignments:
            root.query_type = self.htap_assignments[idx]['query_type']
            root.storage_mode = self.htap_assignments[idx]['storage_mode']
        else:
            root.query_type = 'SELECT'
            root.storage_mode = 'NA'
        root.write_counts = self.per_query_cumulative_counts.get(idx, {'insert': 0, 'update': 0, 'delete': 0}).copy()
        
        root.feature = node2feature(root, encoding, None, None)
        #    print(root)
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
             
                node.query_type = root.query_type
                node.storage_mode = root.storage_mode
             
                node.write_counts = root.write_counts.copy()
                node.parent = root
                root.addChild(node)
        return root

    def calculate_height(self, adj_list,tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:,0]
        child_nodes = adj_list[:,1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order 



def node2feature(node, encoding, hist_file, table_sample):

   
    for k in node.filterDict:
        node.filterDict[k] = node.filterDict[k][:3]
    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((3, 3 - num_filter))
    filts = np.array(list(node.filterDict.values())) #cols, ops, vals
    filts = np.concatenate((filts, pad), axis=1).flatten() 
    mask = np.zeros(3)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId, node.join])
    table = np.array([node.table_id])
    
    feat = np.concatenate((type_join, filts, mask, table))
    return feat
    