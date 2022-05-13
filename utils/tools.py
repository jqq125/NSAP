import numpy as np
import dgl
import torch


class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.iter_counter >= self.num_iterations():
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

def sample_neighbor1(sample_list,samples):
    unique, counts = np.unique(sample_list, return_counts=True)
    total = len(sample_list)
    # p = []
    # for count in counts:
    #     #p += [(count ** (3 / 4)) / count] * count
    #     p += [count / total] * count
    # p = np.array(p)
    # p = p / p.sum()
    samples = min(samples, total)
    # sampled_idx = np.sort(np.random.choice(total, samples, replace=False, p=p))
    sampled_idx = np.sort(np.random.choice(total, samples, replace=False))
    return sampled_idx

def sample_neighbor(sample_list,samples):
    unique, counts = np.unique(sample_list, return_counts=True)
    total = len(sample_list)
    p = []
    for count in counts:
        p += [count / total] * count
    p = np.array(p)
    p = p / p.sum()
    samples = min(samples, total)
    sampled_idx = np.sort(np.random.choice(total, samples, replace=False, p=p))
    return sampled_idx

def parse_adjlist_meta(adjlist, edge_metapath_indices, samples, offset_list=None, mode=None):
    edges = []
    nodes = set()
    result_indices = []
    tmp=sum(offset_list)
    sampled_idx_list=[]
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            sampled_idx = sample_neighbor(row_parsed[1:], samples)
            sampled_idx_list.append(sampled_idx)
            neighbors = [row_parsed[i + 1] for i in sampled_idx]
            result_indices.append(indices[sampled_idx])
        else:
            neighbors = [row_parsed[0]]
            indices = np.array([[row_parsed[0]] * indices.shape[1]])
            if mode == 1:
                indices += offset_list[0]
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping,sampled_idx_list


def parse_minibatch_meta(adjlists_ua, edge_metapath_indices_list_ua, drug_dis_batch, device, samples=None,offset_list=None):
    metaGraph_lists = [[], []]
    result_indices_lists = [[], []]
    idx_batch_mapped_lists = [[], []]
    sampled_idx_lists=[[],[]]
    for mode, (adjlists, edge_metapath_indices_list) in enumerate(zip(adjlists_ua, edge_metapath_indices_list_ua)):
        for adjlist, indices in zip(adjlists, edge_metapath_indices_list):
            edges, result_indices, num_nodes, mapping1,sampled_idx_list= parse_adjlist_meta(
                [adjlist[row[mode]] for row in drug_dis_batch],
                [indices[row[mode]] for row in drug_dis_batch], samples, offset_list, mode)
            sampled_idx_lists[mode].append(sampled_idx_list)
            #获取元图
            metaGraph = dgl.DGLGraph()
            metaGraph=metaGraph.to(device)
            metaGraph.add_nodes(num_nodes)
            if len(edges) > 0:
                sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
                metaGraph.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
                result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
            else:
                result_indices = torch.LongTensor(result_indices).to(device)
            metaGraph_lists[mode].append(metaGraph)
            result_indices_lists[mode].append(result_indices)
            idx_batch_mapped_lists[mode].append(np.array([mapping1[row[mode]] for row in drug_dis_batch]))

    return metaGraph_lists, result_indices_lists, idx_batch_mapped_lists,sampled_idx_lists



def parse_adjlist_context(adjlist, edge_metapath_indices, samples, connection,device, offset_list, mode,sampled_idxs):
    cedges = []
    cnodes= set()
    virtue_idx_lists=[]
    tmp=sum(offset_list)
    first_neighbor=connection[0]

    if mode==0:

        offset1,offset2=sum(offset_list[:2]),0
    if mode==1:
        offset1, offset2 = sum(offset_list[:3]),offset_list[0]

    for row, indices,sampled_idx in zip(adjlist, edge_metapath_indices,sampled_idxs):
        row_parsed = list(map(int, row.split(' ')))

        current_neighbors=set()
        neighbors=[]
        if len(row_parsed) > 1:
            neighbors = [row_parsed[i + 1] for i in sampled_idx]
            for dst in neighbors:
                current_neighbors.add(dst+offset2)
        clists=list(current_neighbors)
        clists.sort()
        virtue_node = tmp
        tmp += 1
        for i, src in enumerate(clists):
            cedges.append((src, virtue_node))


        for dst in clists:
            if dst in cnodes:
                continue
            src_list=first_neighbor[dst-offset2]
            if len(src_list)>0:
                src_idxlist=np.random.choice(len(src_list),min(3,len(src_list)),replace=False)
                src_nlist=src_list[src_idxlist]
                for src in src_nlist:
                    if (src+offset1) not in cnodes:
                        current_neighbors.add(src+offset1)
                    cedges.append((src+offset1,dst))

        virtue_idx_lists.append(virtue_node)
        current_neighbors.add(virtue_node)
        cnodes = cnodes.union(current_neighbors)
        if len(neighbors)==0:
            cnodes.add(row_parsed[0])
            cedges.append((virtue_node,virtue_node))
            cedges.append((row_parsed[0],virtue_node))

    cnodes=list(cnodes)
    cnodes.sort()
    mapping2 = {map_from: map_to for map_to, map_from in enumerate(cnodes)}
    vm_idx_list=[mapping2[virtue_node] for virtue_node in virtue_idx_lists]
    cedges = list(map(lambda tup: (mapping2[tup[0]], mapping2[tup[1]]), cedges))
    src_list = torch.tensor([tup[0] for tup in cedges]).to(device)
    dst_list = torch.tensor([tup[1] for tup in cedges]).to(device)
    g = dgl.graph((src_list, dst_list)).to(device)
    return cedges,cnodes,vm_idx_list,g


def parse_minibatch_context(adjlists_ua, edge_metapath_indices_list_ua, drug_dis_batch, device,connection_list,samples,offset_list,sampled_idx_lists):
    contextGraph_lists = [[], []]
    cnodes_lists=[[],[]]
    vm_idx_lists=[[],[]]
    for mode, (adjlists, edge_metapath_indices_list,connection,sampled_idx_list) in enumerate(zip(adjlists_ua, edge_metapath_indices_list_ua,connection_list,sampled_idx_lists)):
        #print("context mode:", mode)
        for adjlist, indices,sampled_idxs in zip(adjlists, edge_metapath_indices_list,sampled_idx_list):
            # print(len(adjlist))
            #print("t2:", time.time())
            cedges,  cnodes, vm_idx_list,contextGraph = parse_adjlist_context(
                [adjlist[row[mode]] for row in drug_dis_batch],
                [indices[row[mode]] for row in drug_dis_batch], samples, connection,device,offset_list, mode,sampled_idxs)
            #print("t3:", time.time())
            contextGraph_lists[mode].append(contextGraph)
            real_cnodes=cnodes[:-len(drug_dis_batch)]
            cnodes = torch.tensor(real_cnodes).to(device)
            cnodes_lists[mode].append(cnodes)
            vm_idx_lists[mode].append(vm_idx_list)
    # print(g_lists)
    return contextGraph_lists,cnodes_lists,vm_idx_lists



