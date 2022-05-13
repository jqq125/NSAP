import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.pytorch import edge_softmax
import torch.nn.functional as F


class meta_metapath(nn.Module):
    def __init__(self,
                 hidden_dim,
                 num_heads,
                 device,
                 agg_type='neighbor',
                 ):
        super(meta_metapath, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.agg_type = agg_type
        self.device=device

        self.agg = nn.Linear(hidden_dim, num_heads * hidden_dim)



    def message_passing(self, edges):
        m = edges.data['eft'] * edges.data['c_new']
        return {'m': m}

    def forward(self, inputs):

        g, features, type_mask, edge_metapath_indices, target_idx = inputs

        # edata: Instance_number x Seq_len x hidden_dim
        edata = F.embedding(edge_metapath_indices, features)


        if self.agg_type == 'neighbor':
            #Instance_number x hidden_dim
            hidden = edata[:, 0]
            #Instance_number x (hidden_dim*self.num_heads)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            #1 x Instance_number x (hidden_dim*self.num_heads)
            hidden = hidden.unsqueeze(dim=0)
        elif self.agg_type == 'neighbor-linear':
            hidden = self.agg(edata[:, 0])
            hidden = hidden.unsqueeze(dim=0)


        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.hidden_dim)
        c = torch.ones(g.num_edges(),self.num_heads,1).float().to(self.device)

        g.edata.update({'eft': eft, 'c': c})
        g.edata['c_new'] = edge_softmax(g, g.edata.pop('c'))

        g.update_all(self.message_passing, fn.sum('m', 'ft'))
        mft = g.nodes[target_idx].data['ft']
        return mft


