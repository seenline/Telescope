 

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.htap_encoder import HtapEncoderStack
from model.htap_utils import HTAPFeatureExtractor


class Prediction(nn.Module):
    
    def __init__(self, in_feature=69, hid_units=256, contract=1, mid_layers=True, res_con=True):
        super(Prediction, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con
        
        self.out_mlp1 = nn.Linear(in_feature, hid_units)
        self.mid_mlp1 = nn.Linear(hid_units, hid_units//contract)
        self.mid_mlp2 = nn.Linear(hid_units//contract, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, features):
        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = torch.sigmoid(self.out_mlp2(hid))
        return out


class FeatureEmbed(nn.Module):
    
    def __init__(self, embed_size=32, tables=10, types=20, joins=40, columns=30,
                 ops=4, use_sample=True, use_hist=True, bin_number=50):
        super(FeatureEmbed, self).__init__()
        
        self.use_sample = use_sample
        self.embed_size = embed_size        
        self.use_hist = use_hist
        self.bin_number = bin_number
        
        self.typeEmbed = nn.Embedding(types, embed_size)
        self.tableEmbed = nn.Embedding(tables, embed_size)
        self.columnEmbed = nn.Embedding(columns, embed_size)
        self.opEmbed = nn.Embedding(ops, embed_size//8)
        
        self.linearFilter2 = nn.Linear(embed_size+embed_size//8+1, embed_size+embed_size//8+1)
        self.linearFilter = nn.Linear(embed_size+embed_size//8+1, embed_size+embed_size//8+1)
        self.linearType = nn.Linear(embed_size, embed_size)
        self.linearJoin = nn.Linear(embed_size, embed_size)
        self.linearSample = nn.Linear(1000, embed_size)
        self.linearHist = nn.Linear(bin_number, embed_size)
        self.joinEmbed = nn.Embedding(joins, embed_size)
        
        if use_hist:
            self.project = nn.Linear(embed_size*5 + embed_size//8+1, embed_size*5 + embed_size//8+1)
        else:
            self.project = nn.Linear(embed_size*4 + embed_size//8+1, embed_size*4 + embed_size//8+1)
    
    def forward(self, feature):
        typeId, joinId, filtersId, filtersMask, hists, table_sample = torch.split(
            feature, (1, 1, 9, 3, self.bin_number*3, 1001), dim=-1
        )
        
        typeEmb = self.getType(typeId)
        joinEmb = self.getJoin(joinId)
        filterEmbed = self.getFilter(filtersId, filtersMask)
        histEmb = self.getHist(hists, filtersMask)
        tableEmb = self.getTable(table_sample)
    
        if self.use_hist:
            final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb, histEmb), dim=1)
        else:
            final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb), dim=1)
        final = F.leaky_relu(self.project(final))
        
        return final
    
    def getType(self, typeId):
        emb = self.typeEmbed(typeId.long())
        return emb.squeeze(1)
    
    def getTable(self, table_sample):
        table, sample = torch.split(table_sample, (1, 1000), dim=-1)
        emb = self.tableEmbed(table.long()).squeeze(1)
        if self.use_sample:
            emb += self.linearSample(sample)
        return emb
    
    def getJoin(self, joinId):
        emb = self.joinEmbed(joinId.long())
        return emb.squeeze(1)

    def getHist(self, hists, filtersMask):
        histExpand = hists.view(-1, self.bin_number, 3).transpose(1, 2)
        emb = self.linearHist(histExpand)
        emb[~filtersMask.bool()] = 0.
        num_filters = torch.sum(filtersMask, dim=1)
        total = torch.sum(emb, dim=1)
        avg = total / num_filters.view(-1, 1)
        return avg
        
    def getFilter(self, filtersId, filtersMask):
        filterExpand = filtersId.view(-1, 3, 3).transpose(1, 2)
        colsId = filterExpand[:, :, 0].long()
        opsId = filterExpand[:, :, 1].long()
        vals = filterExpand[:, :, 2].unsqueeze(-1)
        
        col = self.columnEmbed(colsId)
        op = self.opEmbed(opsId)
        concat = torch.cat((col, op, vals), dim=-1)
        concat = F.leaky_relu(self.linearFilter(concat))
        concat = F.leaky_relu(self.linearFilter2(concat))
        concat[~filtersMask.bool()] = 0.
        
        num_filters = torch.sum(filtersMask, dim=1)
        total = torch.sum(concat, dim=1)
        avg = total / num_filters.view(-1, 1)
        return avg



class HtapFormer(nn.Module):
    
    
    def __init__(self, 
                 emb_size=32, 
                 ffn_dim=32, 
                 head_size=8,
                 dropout=0.1, 
                 attention_dropout_rate=0.1, 
                 n_layers=8,
                 use_sample=True, 
                 use_hist=True, 
                 bin_number=50,
                 pred_hid=256,
                 storage_feature_dim=8,
                 operator_feature_dim=8,
                 htap_lambda=0.1):
        super(HtapFormer, self).__init__()
        
        if use_hist:
            hidden_dim = emb_size * 5 + emb_size // 8 + 1
        else:
            hidden_dim = emb_size * 4 + emb_size // 8 + 1
            
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.use_sample = use_sample
        self.use_hist = use_hist
        self.storage_feature_dim = storage_feature_dim
        self.operator_feature_dim = operator_feature_dim
        
        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)
        
        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        
        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)
        
        self.embbed_layer = FeatureEmbed(
            emb_size, 
            use_sample=use_sample, 
            use_hist=use_hist, 
            bin_number=bin_number
        )
        
        self.htap_feature_extractor = HTAPFeatureExtractor(
            storage_feature_dim=storage_feature_dim,
            operator_feature_dim=operator_feature_dim
        )
        
        self.input_dropout = nn.Dropout(dropout)
        
        self.encoder = HtapEncoderStack(
            n_layers=n_layers,
            hidden_size=hidden_dim,
            ffn_size=ffn_dim,
            dropout_rate=dropout,
            attention_dropout_rate=attention_dropout_rate,
            head_size=head_size,
            storage_feature_dim=storage_feature_dim,
            operator_feature_dim=operator_feature_dim,
            htap_lambda=htap_lambda
        )
        
        self.pred = Prediction(hidden_dim, pred_hid)
        
        self.pred2 = Prediction(hidden_dim, pred_hid)
        
    def compute_tree_attention_bias(self, attn_bias, rel_pos):
        n_batch = attn_bias.size(0)
        
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1)
        
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2)
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias
        
        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t
        
        return tree_attn_bias
        
    def forward(self, batched_data, htap_data=None):
        attn_bias = batched_data.attn_bias
        rel_pos = batched_data.rel_pos
        x = batched_data.x
        heights = batched_data.heights
        
        n_batch, n_node = x.size()[:2]
        
        tree_attn_bias = self.compute_tree_attention_bias(attn_bias, rel_pos)
        
        x_view = x.view(-1, 1165)
        node_feature = self.embbed_layer(x_view).view(n_batch, -1, self.hidden_dim)
        
        node_feature = node_feature + self.height_encoder(heights)
        
        storage_features = None
        operator_features = None
        
        if htap_data is not None:
            storage_features, operator_features = self.htap_feature_extractor(htap_data)
            
            batch_size = storage_features.size(0)
            zero_storage = torch.zeros(
                batch_size, 1, self.storage_feature_dim, 
                device=storage_features.device
            )
            zero_operator = torch.zeros(
                batch_size, 1, self.operator_feature_dim,
                device=operator_features.device
            )
            
            storage_features = torch.cat([zero_storage, storage_features], dim=1)
            operator_features = torch.cat([zero_operator, operator_features], dim=1)
        
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)
        
        output = self.input_dropout(super_node_feature)
        
        output = self.encoder(
            x=output,
            tree_attn_bias=tree_attn_bias,
            storage_features=storage_features,
            operator_features=operator_features
        )
        
        super_token_output = output[:, 0, :]
        
        cost_pred = self.pred(super_token_output)
        card_pred = self.pred2(super_token_output)
        
        return cost_pred, card_pred


HtapFormerEncoder = HtapFormer


def create_htapformer_model(config):
    model = HtapFormer(
        emb_size=config.get('emb_size', 32),
        ffn_dim=config.get('ffn_dim', 32),
        head_size=config.get('head_size', 8),
        dropout=config.get('dropout', 0.1),
        attention_dropout_rate=config.get('attention_dropout_rate', 0.1),
        n_layers=config.get('n_layers', 8),
        use_sample=config.get('use_sample', True),
        use_hist=config.get('use_hist', True),
        bin_number=config.get('bin_number', 50),
        pred_hid=config.get('pred_hid', 256),
        storage_feature_dim=config.get('storage_feature_dim', 8),
        operator_feature_dim=config.get('operator_feature_dim', 8),
        htap_lambda=config.get('htap_lambda', 0.1)
    )
    return model
