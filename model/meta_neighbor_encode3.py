import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.pytorch import edge_softmax
import torch.nn.functional as F

#生成基于特定元路径的结点向量
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
        #使用多头注意力机制，聚合元路径实例
        self.agg = nn.Linear(hidden_dim, num_heads * hidden_dim)


    #消息函数
    def message_passing(self, edges):
        m = edges.data['eft'] * edges.data['c_new']
        return {'m': m}

    def forward(self, inputs):

        g, features, type_mask, edge_metapath_indices, target_idx = inputs

        # edata: Instance_number x Seq_len x hidden_dim
        #返回元路径实例中对应结点的特征，维度（元路径实例的数量，元路径中结点个数，特征维度）
        edata = F.embedding(edge_metapath_indices, features)

        # 获取基于特定元路径的邻域信息
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

        #eft:基于元路径的邻居信息，维度：Instance_number x num_heads x hidden_dim
        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.hidden_dim)
        c = torch.ones(g.num_edges(),self.num_heads,1).float().to(self.device)


        #为g添加边信息
        g.edata.update({'eft': eft, 'c': c})

        # 计算边的注意力值，默认通过目的结点进行归一化
        g.edata['c_new'] = edge_softmax(g, g.edata.pop('c'))

        # 计算同构邻居的综合结点特征
        g.update_all(self.message_passing, fn.sum('m', 'ft'))
        # 获取同构邻居的综合结点特征，维度：batchsize x num_heads x hidden_dim
        mft = g.nodes[target_idx].data['ft']
        return mft


