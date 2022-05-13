import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.pytorch import edge_softmax
import torch.nn.functional as F
import numpy as np
dim=44336


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

        self.agg = nn.Linear(hidden_dim, num_heads * hidden_dim)


    def forward(self, inputs):
        features, contextGraph, cnodes, vm_idx = inputs


        cnodes=cnodes.long()
        real_ndata = F.embedding(cnodes, features)


        #按结点顺序分配的特征
        virtue_ndata = nn.Embedding(len(vm_idx),self.hidden_dim).weight.to(self.device)

        ndata=torch.cat((real_ndata,virtue_ndata),dim=0)



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

        nft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.hidden_dim)

        c = torch.ones(contextGraph.num_edges(), self.num_heads, 1).float().to(self.device)

        contextGraph.ndata['nft']=nft
        contextGraph.edata['eft']=c

        contextGraph.edata['eft'] = edge_softmax(contextGraph, contextGraph.edata.pop('eft'))


        contextGraph.update_all(fn.u_mul_e('nft','eft','m'),fn.sum('m','new_nft'))
        contextGraph.update_all(fn.u_mul_e('new_nft','eft','m'),fn.sum('m','final_nft'))

        ctft = contextGraph.nodes[vm_idx].data['final_nft']
        return ctft
