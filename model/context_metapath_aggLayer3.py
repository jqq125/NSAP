import torch
import torch.nn as nn
import torch.nn.functional as F
from model.context_neighbor_encode3 import context_metapath


class context_metapath_aggLayer(nn.Module):
    def __init__(self,
                 num_metapaths,
                 hidden_dim,
                 num_heads,
                 batch_size,
                 device
                 ):
        super(context_metapath_aggLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 上下文层
        self.contextgraph_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.contextgraph_layers.append(context_metapath(hidden_dim, num_heads,batch_size,device))

        # metapath-level attention
        self.linear = nn.Parameter(torch.empty(size=(1, num_heads * hidden_dim)))

        # 权重初始化
        nn.init.xavier_normal_(self.linear.data, gain=1.414)

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        # self.fc1 = nn.Linear(hidden_dim * num_heads, attn_vec_dim, bias=True)
        # self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)
        self.fc1 = nn.Linear(hidden_dim * num_heads, 1, bias=False)

        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        # nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):
        # g_list, features, virtue_features, type_mask, edge_metapath_indices_list, target_idx_list, center_node_idx, contextGraph_list, cnodes_list, vm_idx_list = inputs
        g_list, features, type_mask, edge_metapath_indices_list, target_idx_list, center_node_idx, contextGraph_list, cnodes_list, vm_idx_list = inputs

        # metapath-specific layers：生成基于特定元路径的输出Hv,p
        scores = []
        cfts = []
        for contextGraph, cnodes, vm_idx, context_layer in zip(contextGraph_list,cnodes_list,vm_idx_list,self.contextgraph_layers):
            # 返回batch中的结点 基于特定元路径的综合特征,维度：batchsize x (num_heads * hidden_dim)
            cft = context_layer((features, contextGraph, cnodes, vm_idx)).view(-1, self.num_heads * self.hidden_dim)
            cfts.append(cft)

            # q维度：batchsize x (hidden_dim * num_heads)
            q = cft * self.linear

            contextgraph_feat = F.leaky_relu(q)
            # contextgraph_feat维度：batchsize x (num_heads * hidden_dim)
            contextgraph_feat = torch.mean(contextgraph_feat, dim=0)
            attn_score = self.fc1(contextgraph_feat)
            scores.append(attn_score)

        scores = torch.cat(scores, dim=0)
        scores = F.softmax(scores, dim=0)
        scores = torch.unsqueeze(scores, dim=-1)
        scores = torch.unsqueeze(scores, dim=-1)
        cfts = [torch.unsqueeze(cft, dim=0) for cft in cfts]
        cfts = torch.cat(cfts, dim=0)
        context_h = torch.sum(scores * cfts, dim=0)
        return context_h