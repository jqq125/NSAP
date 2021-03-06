import pickle
import numpy as np
from scipy import sparse


def load_BioNet_data(save_prefix='./dataset/preprocess_NSAP'):
    in_file = open(save_prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx_usu = pickle.load(in_file)
    in_file.close()

    in_file = open(save_prefix + '/0/0-2-0_idx.pickle', 'rb')
    idx_utu = pickle.load(in_file)
    in_file.close()

    in_file = open(save_prefix + '/1/1-0-1_idx.pickle', 'rb')
    idx_sus = pickle.load(in_file)
    in_file.close()

    in_file = open(save_prefix + '/1/1-3-1_idx.pickle', 'rb')
    idx_sgs = pickle.load(in_file)
    in_file.close()

    infile = open(save_prefix + '/0/0-1-0.adjlist', 'r')
    adj_usu = [line.strip() for line in infile]
    infile.close()

    infile = open(save_prefix + '/0/0-2-0.adjlist', 'r')
    adj_utu = [line.strip() for line in infile]
    infile.close()

    infile = open(save_prefix + '/1/1-0-1.adjlist', 'r')
    adj_sus = [line.strip() for line in infile]
    infile.close()

    infile = open(save_prefix + '/1/1-3-1.adjlist', 'r')
    adj_sgs = [line.strip() for line in infile]
    infile.close()

    connection=np.load(save_prefix+'/connection.npz',allow_pickle=True)
    ut=connection['ut'].item()
    sg=connection['sg'].item()


    drug_adj1, drug_idx1 = adj_utu, idx_utu
    drug_adj2, drug_idx2 = adj_usu, idx_usu
    dis_adj1, dis_idx1 = adj_sgs, idx_sgs
    dis_adj2, dis_idx2 = adj_sus, idx_sus


    adjlist_ua = [[drug_adj1,drug_adj2],[dis_adj1,dis_adj2]]
    idxlist_ua = [[drug_idx1, drug_idx2], [dis_idx1, dis_idx2]]

    #adjM = sparse.load_npz(save_prefix + '/adjM.npz')
    type_mask = np.load(save_prefix + '/node_types_NSAP.npy')
    train_val_test_pos_drug_dis = np.load(save_prefix + '/train_val_test_pos_drug_dis.npz')
    train_val_test_neg_drug_dis = np.load(save_prefix + '/train_val_test_neg_drug_dis.npz')

    connection_list=[[ut],[sg]]
    return connection_list,adjlist_ua, idxlist_ua, type_mask, train_val_test_pos_drug_dis, train_val_test_neg_drug_dis
