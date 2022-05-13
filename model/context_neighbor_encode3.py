import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.pytorch import edge_softmax
import torch.nn.functional as F
import numpy as np
dim=44336

# 生成基于特定元路径的结点向量
class context_metapath(nn.Module):
    def __init__(self,
                 hidden_dim,
                 num_heads,
                 batch_size,
                 device,
                 agg_type='neighbor'
                 ):
        super(context_metapath, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.device=device
        self.batch_size=batch_size
        self.agg_type = agg_type
        # 使用多头注意力机制，聚合元路径实例
        self.agg = nn.Linear(hidden_dim, num_heads * hidden_dim)


    def forward(self, inputs):
        features, contextGraph, cnodes, vm_idx = inputs

        # 上下文图聚合
        # 结点特征获取,维度:结点个数 x 特征维度
        cnodes=cnodes.long()
        real_ndata = F.embedding(cnodes, features)


        #按结点顺序分配的特征
        virtue_ndata = nn.Embedding(len(vm_idx),self.hidden_dim).weight.to(self.device)

        ndata=torch.cat((real_ndata,virtue_ndata),dim=0)


        # 聚合基于特定元路径的邻域信息
        if self.agg_type == 'neighbor':
            # node_number x hidden_dim
            hidden = ndata
            # node_number x (hidden_dim*self.num_heads)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            # 1 x node_number x (hidden_dim*self.num_heads)
            hidden = hidden.unsqueeze(dim=0)
        elif self.agg_type == 'neighbor-linear':
            hidden = self.agg(ndata)
            hidden = hidden.unsqueeze(dim=0)

        # nft:基于元路径的邻居信息，维度：node_number x num_heads x hidden_dim
        nft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.hidden_dim)

        c = torch.ones(contextGraph.num_edges(), self.num_heads, 1).float().to(self.device)

        # c=contextGraph.edata.pop('eft')
        # c = torch.cat([c] * self.num_heads, dim=1)
        # c = c.unsqueeze(dim=0)
        # 为g的结点和边添加特征信息
        contextGraph.ndata['nft']=nft
        contextGraph.edata['eft']=c
        # 计算边的注意力值，默认通过目的结点进行归一化
        # 计算边的注意力值，默认通过目的结点进行归一化
        contextGraph.edata['eft'] = edge_softmax(contextGraph, contextGraph.edata.pop('eft'))


        # 计算同构邻居的综合结点特征
        contextGraph.update_all(fn.u_mul_e('nft','eft','m'),fn.sum('m','new_nft'))
        contextGraph.update_all(fn.u_mul_e('new_nft','eft','m'),fn.sum('m','final_nft'))
        # 获取同构邻居的综合结点特征，维度：batchsize x num_heads x hidden_dim
        ctft = contextGraph.nodes[vm_idx].data['final_nft']
        return ctft