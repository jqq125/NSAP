import os
import time

import torch
import random
import numpy as np
import torch.nn.functional as F
from utils.parser import parse_args
from utils.data_loader import load_BioNet_data
from utils.helper import EarlyStopping
from utils.tools import index_generator, parse_minibatch_meta,parse_minibatch_context
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import lr_scheduler


num_drug=1482
num_dis=793
num_target=2077
num_gene=6365
offset_list=[num_drug,num_dis,num_target,num_gene]


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed==2022:
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    """fix the random seed"""
    init_seed(2021)

    """read args"""
    global args, device
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if (args.cuda and torch.cuda.is_available()) else torch.device("cpu")
    #device = torch.device("cpu")
    neighbor_samples = args.samples
    #sampling_ratio = args.Sampling_ratio
    save_postfix = args.save_postfix


    """build dataset"""
    connection_list,adjlists_ua, edge_metapath_indices_list_ua, type_mask, train_val_test_pos_drug_dis, train_val_test_neg_drug_dis = load_BioNet_data()
    features_list = []
    in_dims = []

    """Initilize node features"""
    # one-hot vector used to node features
    for i in range(args.num_ntype):
        dim = (type_mask == i).sum()
        in_dims.append(dim)
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(dim))
        features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))

    """get the pos smaples and neg samples of the train/val/test set"""
    train_pos_drug_dis = train_val_test_pos_drug_dis['train_pos_drug_dis']
    val_pos_drug_dis = train_val_test_pos_drug_dis['val_pos_drug_dis']
    test_pos_drug_dis = train_val_test_pos_drug_dis['test_pos_drug_dis']
    train_neg_drug_dis = train_val_test_neg_drug_dis['train_neg_drug_dis']
    val_neg_drug_dis = train_val_test_neg_drug_dis['val_neg_drug_dis']
    test_neg_drug_dis = train_val_test_neg_drug_dis['test_neg_drug_dis']

    """the tested result set's flag"""
    y_true_test = np.array([1] * len(test_pos_drug_dis) + [0] * len(test_neg_drug_dis))

    auc_list = []
    ap_list = []

    """define model"""
    from model.NSAP_lp3 import NSAP_lp
    for i in range(args.repeat):
    #for i in range(1):
        model = NSAP_lp(
            num_metapaths_list=[2, 2],
            feats_dim_list=in_dims,
            hidden_dim=args.hidden_size,
            out_dim=args.hidden_size,
            num_heads=args.num_heads,
            batch_size=args.batch_size,
            device=device,
            dropout_rate=args.dropout_rate,
            offset=offset_list[0]
        ).to(device)

        """define optimizer"""
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5,15,20], gamma=0.1)

        """Train Model"""

        stopper = EarlyStopping(patience=args.patience, verbose=True, delta=0, save_path=f'checkpoint/checkpoint_{save_postfix}_{args.num_heads}_{args.samples}_{i}.pt')
        #
        print("start training ...")
        dur1 = []
        dur2 = []
        dur3 = []
        train_pos_idx_generator = index_generator(batch_size=args.batch_size, num_data=len(train_pos_drug_dis))
        val_idx_generator = index_generator(batch_size=args.batch_size, num_data=len(val_pos_drug_dis), shuffle=False)
        # train_eloss_lists = []
        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            model.train()
            loss = 0
            for iteration in range(train_pos_idx_generator.num_iterations()):
                # forward
                t0 = time.time()
                train_pos_idx_batch = train_pos_idx_generator.next()
                train_pos_idx_batch.sort()
                train_pos_drug_dis_batch = train_pos_drug_dis[train_pos_idx_batch].tolist()
                train_neg_idx_batch = np.random.choice(len(train_neg_drug_dis), len(train_pos_idx_batch))
                # train_neg_idx_batch = np.random.choice(len(train_neg_drug_dis), sampling_ratio*len(train_pos_idx_batch))
                train_neg_idx_batch.sort()
                train_neg_drug_dis_batch = train_neg_drug_dis[train_neg_idx_batch].tolist()

                # shuffle
                num_pos = train_pos_idx_batch.shape[0]
                train_batch = np.concatenate([train_pos_drug_dis_batch, train_neg_drug_dis_batch], axis=0)
                y_label = np.zeros((train_batch.shape[0], 1), dtype=int)
                y_label[:num_pos] = 1
                train_data = np.concatenate([train_batch, y_label], axis=1)
                np.random.shuffle(train_data)
                train_batch = train_data[:, :-1]
                y_label = train_data[:, -1]


                t00=time.time()
                train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists,train_pos_batch_sampled_idx_lists = parse_minibatch_meta(
                    adjlists_ua, edge_metapath_indices_list_ua, train_pos_drug_dis_batch, device,
                    neighbor_samples, offset_list)
                # t01 = time.time()
                # print("cur1:", t01 - t00)
                train_pos_contextGraph_lists,train_pos_cnodes_lists,train_pos_vm_idx_lists = parse_minibatch_context(
                    adjlists_ua, edge_metapath_indices_list_ua, train_pos_drug_dis_batch, device,connection_list,
                    neighbor_samples, offset_list, train_pos_batch_sampled_idx_lists)
                # t02 = time.time()
                # print("cur2:", t02 - t01)
                train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists,batch_sampled_idx_lists = parse_minibatch_meta(
                    adjlists_ua, edge_metapath_indices_list_ua, train_neg_drug_dis_batch, device,
                    neighbor_samples, offset_list)
                # t03 = time.time()
                # print("cur3:", t03 - t02)
                train_neg_contextGraph_lists,train_neg_cnodes_lists,train_neg_vm_idx_lists = parse_minibatch_context(
                    adjlists_ua, edge_metapath_indices_list_ua, train_neg_drug_dis_batch, device,connection_list,
                    neighbor_samples,offset_list,batch_sampled_idx_lists)


                t1 = time.time()
                dur1.append(t1 - t0)
                # print("cur4:", t1 - t03)

                [pos_embedding_drug, pos_embedding_dis] = model(
                    (train_pos_g_lists, features_list, type_mask, train_pos_indices_lists,train_pos_idx_batch_mapped_lists,
                     train_pos_drug_dis_batch,train_pos_contextGraph_lists,train_pos_cnodes_lists,train_pos_vm_idx_lists))
                [neg_embedding_drug, neg_embedding_dis] = model(
                    (train_neg_g_lists, features_list, type_mask, train_neg_indices_lists,train_neg_idx_batch_mapped_lists,
                     train_neg_drug_dis_batch,train_neg_contextGraph_lists,train_neg_cnodes_lists,train_neg_vm_idx_lists ))


                pos_embedding_drug = pos_embedding_drug.view(-1, 1, pos_embedding_drug.shape[1])
                pos_embedding_dis = pos_embedding_dis.view(-1, pos_embedding_dis.shape[1], 1)
                neg_embedding_drug = neg_embedding_drug.view(-1, 1, neg_embedding_drug.shape[1])
                neg_embedding_dis = neg_embedding_dis.view(-1, neg_embedding_dis.shape[1], 1)
                pos_out = torch.bmm(pos_embedding_drug, pos_embedding_dis)
                neg_out = -torch.bmm(neg_embedding_drug, neg_embedding_dis)
                train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
                # train_loss = -(torch.sum((1 - alpha) * F.logsigmoid(pos_out)) +
                #                         torch.sum(alpha * F.logsigmoid(neg_out)))

                t2 = time.time()
                dur2.append(t2 - t1)
                # loss+=train_loss

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()


                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info

                if iteration % 50 == 0:
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))

            # train_eloss_lists.append(loss)
            # train_endtime=time.time()
            # print('Epoch {:05d} | Train_Loss {:.4f} | Time(s) {:.4f}'.format(
            #     epoch, loss.item(), train_endtime - t_start))
            # validation
            model.eval()
            val_loss = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_pos_drug_dis_batch = val_pos_drug_dis[val_idx_batch].tolist()
                    val_neg_drug_dis_batch = val_neg_drug_dis[val_idx_batch].tolist()
                    val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists,val_pos_batch_sampled_idx_lists = parse_minibatch_meta(
                        adjlists_ua, edge_metapath_indices_list_ua, val_pos_drug_dis_batch, device,
                        neighbor_samples, offset_list)
                    val_pos_contextGraph_lists, val_pos_cnodes_lists, val_pos_vm_idx_lists = parse_minibatch_context(
                        adjlists_ua, edge_metapath_indices_list_ua, val_pos_drug_dis_batch, device, connection_list,
                        neighbor_samples, offset_list,val_pos_batch_sampled_idx_lists)

                    val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists,val_neg_batch_sampled_idx_lists = parse_minibatch_meta(
                        adjlists_ua, edge_metapath_indices_list_ua, val_neg_drug_dis_batch, device,
                        neighbor_samples, offset_list)
                    val_neg_contextGraph_lists, val_neg_cnodes_lists, val_neg_vm_idx_lists = parse_minibatch_context(
                        adjlists_ua, edge_metapath_indices_list_ua, val_neg_drug_dis_batch, device, connection_list,
                        neighbor_samples, offset_list,val_neg_batch_sampled_idx_lists)

                    [pos_embedding_drug, pos_embedding_dis] = model(
                        (val_pos_g_lists, features_list, type_mask, val_pos_indices_lists,
                         val_pos_idx_batch_mapped_lists,
                         val_pos_drug_dis_batch, val_pos_contextGraph_lists, val_pos_cnodes_lists,
                         val_pos_vm_idx_lists))
                    [neg_embedding_drug, neg_embedding_dis] = model(
                        (val_neg_g_lists, features_list, type_mask, val_neg_indices_lists,
                         val_neg_idx_batch_mapped_lists,
                         val_neg_drug_dis_batch, val_neg_contextGraph_lists, val_neg_cnodes_lists,
                         val_neg_vm_idx_lists))

                    pos_embedding_drug = pos_embedding_drug.view(-1, 1, pos_embedding_drug.shape[1])
                    pos_embedding_dis = pos_embedding_dis.view(-1, pos_embedding_dis.shape[1], 1)
                    neg_embedding_drug = neg_embedding_drug.view(-1, 1, neg_embedding_drug.shape[1])
                    neg_embedding_dis = neg_embedding_dis.view(-1, neg_embedding_dis.shape[1], 1)

                    pos_out = torch.bmm(pos_embedding_drug, pos_embedding_dis)
                    neg_out = -torch.bmm(neg_embedding_drug, neg_embedding_dis)
                    val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)))
                val_loss = torch.mean(torch.tensor(val_loss))
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping,返回loss是否下降
            flag=stopper(val_loss, model)
            scheduler.step()
            print("the learning rate of epoch %d:%f" % (epoch+1, optimizer.param_groups[0]['lr']))
            if stopper.early_stop:
                print('Early stopping!')
                break

        # np.save(f'record/epochloss.npy',train_eloss_lists)
        print('start testing')
        test_idx_generator = index_generator(batch_size=args.batch_size, num_data=len(test_pos_drug_dis), shuffle=False)
        stopper.load_checkpoint(model, ckp_path=f'checkpoint/checkpoint_{save_postfix}_{args.num_heads}_{args.samples}_{i}.pt')
        model.eval()
        pos_proba_list = []
        neg_proba_list = []
        drug_embedding_list = []
        dis_embedding_list = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_pos_drug_dis_batch = test_pos_drug_dis[test_idx_batch].tolist()
                test_neg_drug_dis_batch = test_neg_drug_dis[test_idx_batch].tolist()
                test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists,test_pos_batch_sampled_idx_lists = parse_minibatch_meta(
                    adjlists_ua, edge_metapath_indices_list_ua, test_pos_drug_dis_batch, device,
                    neighbor_samples, offset_list)
                test_pos_contextGraph_lists, test_pos_cnodes_lists, test_pos_vm_idx_lists = parse_minibatch_context(
                    adjlists_ua, edge_metapath_indices_list_ua, test_pos_drug_dis_batch, device, connection_list,
                    neighbor_samples, offset_list,test_pos_batch_sampled_idx_lists)

                test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists,test_neg_batch_sampled_idx_lists = parse_minibatch_meta(
                    adjlists_ua, edge_metapath_indices_list_ua, test_neg_drug_dis_batch, device,
                    neighbor_samples, offset_list)
                test_neg_contextGraph_lists, test_neg_cnodes_lists, test_neg_vm_idx_lists = parse_minibatch_context(
                    adjlists_ua, edge_metapath_indices_list_ua, test_neg_drug_dis_batch, device, connection_list,
                    neighbor_samples, offset_list,test_neg_batch_sampled_idx_lists)

                [pos_embedding_drug, pos_embedding_dis] = model(
                    (
                    test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists,
                    test_pos_drug_dis_batch, test_pos_contextGraph_lists, test_pos_cnodes_lists,
                    test_pos_vm_idx_lists)
                    )
                [neg_embedding_drug, neg_embedding_dis] = model(
                    (
                    test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists,
                    test_neg_drug_dis_batch, test_neg_contextGraph_lists, test_neg_cnodes_lists,
                    test_neg_vm_idx_lists)
                    )

                pos_embedding_drug = pos_embedding_drug.view(-1, 1, pos_embedding_drug.shape[1])
                pos_embedding_dis = pos_embedding_dis.view(-1, pos_embedding_dis.shape[1], 1)
                neg_embedding_drug = neg_embedding_drug.view(-1, 1, neg_embedding_drug.shape[1])
                neg_embedding_dis = neg_embedding_dis.view(-1, neg_embedding_dis.shape[1], 1)

                drug_embedding_list.append(pos_embedding_drug.squeeze().cpu().numpy())
                dis_embedding_list.append(pos_embedding_dis.squeeze().cpu().numpy())

                pos_out = torch.bmm(pos_embedding_drug, pos_embedding_dis).flatten()
                neg_out = torch.bmm(neg_embedding_drug, neg_embedding_dis).flatten()
                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))
            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().numpy()
        np.savez(r'./record/NSAP_prediction_result.npz', y_true=y_true_test, y_pred=y_proba_test)
        np.savez(r'./record/NSAP_embedding2vis_InfoMaxFusv1.npz', drug=np.concatenate(drug_embedding_list, axis=0),
                                    disease=np.concatenate(dis_embedding_list, axis=0))

        auc = roc_auc_score(y_true_test, y_proba_test)
        ap = average_precision_score(y_true_test, y_proba_test)
        print('Link Prediction Test')
        print('AUC = {}'.format(auc))
        print('AP = {}'.format(ap))
        auc_list.append(auc)
        ap_list.append(ap)

    print('----------------------------------------------------------------')
    print('Link Prediction Tests Summary')
    print('AUC_list={},AUC_mean = {}, AUC_std = {}'.format(auc_list,np.mean(auc_list), np.std(auc_list)))
    print('AP_list={},AP_mean = {}, AP_std = {}'.format(ap_list,np.mean(ap_list), np.std(ap_list)))

