import torch
import torch.nn as nn
import numpy as np
from model.NSAP_lp_layer3 import  NSAP_lp_layer

class NSAP_lp(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 batch_size,
                 device,
                 dropout_rate,
                 offset):
        super(NSAP_lp, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size=batch_size
        self.offset=offset

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)


        # NSAP_lp layers
        self.layer = NSAP_lp_layer(num_metapaths_list,
                                     hidden_dim,
                                     out_dim,
                                     num_heads,
                                     batch_size,
                                     device
                                   )


    def forward(self, inputs):
        #g_lists, features_list,virtue_features_list, type_mask, edge_metapath_indices_lists, target_idx_lists,train_batch,contextGraph_lists,cnodes_lists,vm_idx_lists= inputs
        g_lists, features_list,  type_mask, edge_metapath_indices_lists, target_idx_lists, train_batch, contextGraph_lists, cnodes_lists, vm_idx_lists = inputs

        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        transformed_features = self.feat_drop(transformed_features)

        [logits_drug, logits_dis]= self.layer(
            (g_lists, transformed_features,type_mask, edge_metapath_indices_lists, target_idx_lists, train_batch,contextGraph_lists,cnodes_lists,vm_idx_lists,self.offset))

        return [logits_drug, logits_dis]