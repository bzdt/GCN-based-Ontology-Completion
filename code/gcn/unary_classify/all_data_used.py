from dgl import DGLGraph
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from layers import RGCNBasisLayer as RGCNLayer
from model import BaseRGCN
from utils import find_rules_ut_all, find_same_etype
import random
from data_preprocessing import load_whole_data
from itertools import combinations
import warnings
import os

warnings.filterwarnings('ignore')

# random.seed(123)

class UnaryClassify(BaseRGCN):
    def create_features(self):
        features = torch.arange(self.in_feat)
        if self.use_cuda:
            featurs = features.cuda()
        return features

    def build_input_layer(self):
        return RGCNLayer(self.in_feat, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True, node_features=self.features)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, dropout=self.dropout,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=None)


# similar loss
def similar_loss(g, node_idx, logits, direction='in'):
    all_loss = 0
    for i in node_idx:
        if direction == 'in':
            u, _, eid = g.in_edges(i, 'all')
        else:
            _, u, eid = g.out_edges(i, 'all')
        etype = g.edata['type'][eid]
        loss = 0
        for idx in find_same_etype(etype):
            sm_nodes = u[idx[1]]
            cnt = 0
            l = 0
            for n1, n2 in combinations(sm_nodes, 2):
                cnt += 1
                l += torch.norm(logits[n1] - logits[n2], 1)
            loss = l / cnt
        all_loss += loss
    return all_loss / len(node_idx)


def main(args):
    fold = 10
    topk = 100

    if not os.path.exists('result/'):
        os.mkdir('result/')

    if not os.path.exists('rules/'):
        os.mkdir('rules/')

    sigma = args.sigma
    beta = args.beta
    path = 'dataset/' + args.dataset + '/' + str(fold) + '_fold' + '/'
    pred_rules_out = ''
    num_node, edge_list, edge_src, edge_dst, edge_type, edge_norm, num_rel, train_idx, data_label, node_features = load_whole_data(
                path, args.ftype, args.n_hidden)

    train_label = data_label
    test_label = data_label

    tmp_label = train_label
    train_idx = list(train_idx)
    test_idx = train_idx

    edge_type = torch.from_numpy(edge_type).long()
    edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)
    node_features = torch.from_numpy(np.array(node_features)).float()

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        edge_type = edge_type.cuda()
        edge_norm = edge_norm.cuda()
        node_features = node_features.cuda()

    # create multi-graph
    g = DGLGraph(multigraph=True)
    g.add_nodes(num_node)
    g.add_edges(edge_src, edge_dst)
    g.edata.update({'type': edge_type, 'norm': edge_norm})

    # node_norm for apply_func
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    node_norm = torch.from_numpy(norm)
    if use_cuda:
        node_norm = node_norm.cuda()
    g.ndata.update({'norm': node_norm})

    in_feat = node_features.shape[1]
    num_classes = train_label.shape[1]

    train_label = torch.from_numpy(np.array(train_label)).float()
    test_label = torch.from_numpy(np.array(test_label)).float()
    # create model
    model = UnaryClassify(in_feat,
                            args.n_hidden,
                            num_classes,
                            num_rel,
                            num_bases=args.n_bases,
                            num_hidden_layers=args.n_layers - 2,
                            dropout=args.dropout,
                            use_cuda=use_cuda,
                            features=node_features)
    if use_cuda:
        model.cuda()
        train_label.cuda()
        test_label.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')  #

    if args.validation:
        val_idx = sorted(random.sample(list(train_idx), num_node // 5))  # 10% for validation
        train_idx = sorted(list(set(train_idx) - set(val_idx)))
    else:
        val_idx = train_idx
    val_labels = train_label[val_idx]
    train_label = train_label[train_idx]
    print("start training...")
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        logits = model.forward(g)
        # similar loss
        sim_loss_in = similar_loss(g, train_idx, logits, 'in')
        sim_loss_out = similar_loss(g, train_idx, logits, 'out')
        classify_loss = criterion(logits[train_idx], train_label)  # for multi-label classification
        loss = classify_loss + sigma * sim_loss_in + beta * sim_loss_out

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()


    model.eval()
    logits = model.forward(g).cpu()
    logits = F.sigmoid(logits)

    test_prob = logits.detach().numpy()

    str_prerule = find_rules_ut_all(path, tmp_label, test_prob, topk)

    pred_rules_out += str_prerule

    file_name = args.ftype + '_' + args.dataset + '_all_data.txt'

    f = open('rules/pred_' + file_name, 'w', encoding='utf-8')
    f.write(pred_rules_out)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=32,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
                        help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, default='wine',
                       help="dataset to use")
    parser.add_argument("-f", "--ftype", type=str, default='embedding',
                       help="feature type to use")
    parser.add_argument("--l2norm", type=float, default=0,
                        help="l2 norm coef")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--sigma", type=float, default=0.01,
                        help="similar loss in coef")
    parser.add_argument("--beta", type=float, default=0.01,
                        help="similar loss out coef")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    args.bfs_level = 0
    main(args)