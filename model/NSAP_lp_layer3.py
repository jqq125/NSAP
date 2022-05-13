import torch
import torch.nn as nn
from  model.meta_metapath_aggLayer3 import meta_metapath_aggLayer
from model.context_metapath_aggLayer3 import context_metapath_aggLayer
# for link prediction task
class NSAP_lp_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 batch_size,
                 device
                ):
        super(NSAP_lp_layer, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.batch_size=batch_size
        self.device=device


        self.drug_layers=nn.ModuleList()
        self.drug_layers.append(meta_metapath_aggLayer(num_metapaths_list[0],hidden_dim,num_heads,device))
        self.drug_layers.append(context_metapath_aggLayer(num_metapaths_list[0],hidden_dim,num_heads,batch_size,device))

        self.dis_layers = nn.ModuleList()
        self.dis_layers.append(meta_metapath_aggLayer(num_metapaths_list[1], hidden_dim, num_heads,device))
        self.dis_layers.append(context_metapath_aggLayer(num_metapaths_list[1],hidden_dim,num_heads,batch_size,device))

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc_drugs=nn.ModuleList([nn.Linear(hidden_dim * num_heads, out_dim, bias=True) for _ in range(2)])
        self.fc_diss=nn.ModuleList([nn.Linear(hidden_dim * num_heads, out_dim, bias=True) for _ in range(2)])
        for fc_drug,fc_dis in zip(self.fc_drugs,self.fc_diss):
            nn.init.xavier_normal_(fc_drug.weight, gain=1.414)
            nn.init.xavier_normal_(fc_dis.weight, gain=1.414)

        self.linear = nn.Parameter(torch.empty(size=(2, 1)))
        nn.init.xavier_normal_(self.linear.data, gain=1.414)


    def forward(self, inputs):
        # g_lists, features,virtue_features, type_mask, edge_metapath_indices_lists, target_idx_lists,train_batch ,contextGraph_lists,cnodes_lists,vm_idx_lists= inputs
        g_lists, features, type_mask, edge_metapath_indices_lists, target_idx_lists, train_batch, contextGraph_lists, cnodes_lists, vm_idx_lists,offset = inputs

        # train_batch = torch.tensor(train_batch).cuda()
        train_batch = torch.tensor(train_batch).to(self.device)
        train_drug_batch=train_batch[:,0]
        train_dis_batch=train_batch[:,1]+offset

        for i,(drug_layer,fc_drug,dis_layer,fc_dis) in enumerate(zip(self.drug_layers,self.fc_drugs,self.dis_layers,self.fc_diss)):
            # if i==1:
            #     break
            h_drug = drug_layer(
                (g_lists[0], features, type_mask, edge_metapath_indices_lists[0], target_idx_lists[0], train_drug_batch,contextGraph_lists[0],cnodes_lists[0],vm_idx_lists[0]))
            logits_drug = fc_drug(h_drug)
            h_dis = dis_layer(
                (g_lists[1], features, type_mask, edge_metapath_indices_lists[1], target_idx_lists[1], train_dis_batch,contextGraph_lists[1],cnodes_lists[1],vm_idx_lists[1]))
            logits_dis = fc_dis(h_dis)
            if i==0:
                meta_emb_list=torch.stack((logits_drug,logits_dis),dim=0)
            else:
                context_emb_list=torch.stack((logits_drug,logits_dis),dim=0)
        h_lists=torch.stack((meta_emb_list,context_emb_list),dim=1)
        h_lists=h_lists.permute(0, 2, 1, 3)
        emb_lists=torch.sum(h_lists * self.linear,dim=2)
        return emb_lists
