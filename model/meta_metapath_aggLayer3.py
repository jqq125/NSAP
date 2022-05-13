import torch
import torch.nn as nn
import torch.nn.functional as F
from model.meta_neighbor_encode3 import meta_metapath

class meta_metapath_aggLayer(nn.Module):
    def __init__(self,
                 num_metapaths,
                 hidden_dim,
                 num_heads,
                 device
                 ):
        super(meta_metapath_aggLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads


        self.metagraph_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.metagraph_layers.append(meta_metapath(hidden_dim, num_heads,device))

        # metapath-level attention
        self.linear_dst = nn.Parameter(torch.empty(size=(1, num_heads*hidden_dim)))
        self.linear_src = nn.Parameter(torch.empty(size=(1, num_heads*hidden_dim)))

        nn.init.xavier_normal_(self.linear_dst.data, gain=1.414)
        nn.init.xavier_normal_(self.linear_src.data, gain=1.414)


        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc1 = nn.Linear(hidden_dim * num_heads, 1, bias=False)



        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        # nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):
        g_list, features, type_mask, edge_metapath_indices_list, target_idx_list ,center_node_idx,contextGraph_list,cnodes_list,vm_idx_list= inputs


        scores=[]
        mfts=[]
        for g, edge_metapath_indices, target_idx,meta_layer in zip(g_list, edge_metapath_indices_list, target_idx_list,self.metagraph_layers):

            mft=meta_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1,self.num_heads * self.hidden_dim)
            mfts.append(mft)


            center_node_feat = F.embedding(center_node_idx, features)
            center_node_feat = torch.cat([center_node_feat] * self.num_heads, dim=1)

    
            r1 = center_node_feat * self.linear_dst
            r2 = mft * self.linear_src

            metagraph_feat = F.leaky_relu(r1+r2)
            metagraph_feat=torch.mean(metagraph_feat, dim=0)
            attn_score = self.fc1(metagraph_feat)
            scores.append(attn_score)


        scores = torch.cat(scores, dim=0)
        scores = F.softmax(scores, dim=0)
        scores = torch.unsqueeze(scores, dim=-1)
        scores = torch.unsqueeze(scores, dim=-1)
        mfts = [torch.unsqueeze(mft, dim=0) for mft in mfts]
        mfts = torch.cat(mfts, dim=0)
        meta_h = torch.sum(scores * mfts, dim=0)
        return meta_h
