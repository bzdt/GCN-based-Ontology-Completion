"""
Modeling Relational Data with Graph Convolutional Networks
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import warnings
import os

from layers import RGCNBasisLayer as RGCNLayer
from model import BaseRGCN

from data_preprocessing import train_test_triples

import utils
import numpy as np

warnings.filterwarnings('ignore')


class RGCN(BaseRGCN):
    def build_input_layer(self):
        return RGCNLayer(self.in_feat, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True, node_features=self.features)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=act, dropout=self.dropout)


class BinaryPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0, node_features=None, relation_features=None):
        super(BinaryPredict, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda, features=node_features)
        self.reg_param = reg_param
        if relation_features is None:
            self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
            nn.init.xavier_uniform_(self.w_relation,
                                    gain=nn.init.calculate_gain('relu'))
        else:
            self.w_relation = nn.Parameter(torch.Tensor(relation_features))


    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0].long()]
        r = self.w_relation[triplets[:,1].long()]
        o = embedding[triplets[:,2].long()]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g):
        return self.rgcn.forward(g)

    def evaluate(self, g):
        # get embedding and relation weight without grad
        embedding = self.forward(g)
        return embedding, self.w_relation

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, triplets, labels):
        embedding = self.forward(g)
        score = self.calc_score(embedding, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embedding)
        return predict_loss + self.reg_param * reg_loss


def main(args):
    if not os.path.exists('result/'):
        os.mkdir('result/')

    if not os.path.exists('rules/'):
        os.mkdir('rules/')

    results = []
    fold = 10
    results_out = str(args) + '\n'
    true_rules_out = ''
    pred_rules_out = ''
    for i in range(fold):
        path = 'dataset/' + args.dataset + '/' + str(fold) + '_fold/'
        train_data, test_data, num_nodes, num_rels, node_features = train_test_triples(path, i, args.ftype, args.n_hidden)
        complete_data = np.concatenate((train_data, test_data))
        print("Edges: ", len(complete_data))

        if test_data.shape[0] == 0:
            continue

        # create model
        in_feat = node_features.shape[1]
        node_features = torch.from_numpy(np.array(node_features)).float()

        # check cuda
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(args.gpu)

        if use_cuda:
            node_features = node_features.cuda()

        model_state_file = args.dataset + '_' + args.ftype + '_model_state.pth'

        model = BinaryPredict(in_feat,
                              args.n_hidden,
                              num_rels,
                              num_bases=args.n_bases,
                              num_hidden_layers=args.n_layers,
                              dropout=args.dropout,
                              use_cuda=use_cuda,
                              reg_param=args.regularization,
                              node_features=node_features,
                              relation_features=None)
        if use_cuda:
            model.cuda()

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # validation and testing triples
        train_idx = np.arange(len(train_data))
        valid_idx = sorted(random.sample(list(train_idx), len(train_idx) // 5))
        train_idx = sorted(list(set(train_idx) - set(valid_idx)))
        valid_data = train_data[valid_idx]
        train_data = train_data[train_idx]
        valid_data = torch.LongTensor(valid_data)
        test_data = torch.LongTensor(test_data)

        # build test graph
        test_graph, test_rel, test_norm, edge_norm = utils.build_test_graph(
            num_nodes, num_rels, complete_data)
        test_node_id = torch.arange(0, num_nodes, dtype=torch.long)
        test_rel = torch.from_numpy(test_rel)
        edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)

        test_rel = test_rel.long()
        test_norm = torch.from_numpy(test_norm)
        if use_cuda:
            test_node_id = test_node_id.cuda()
            test_rel, test_norm = test_rel.cuda(), test_norm.cuda()
            edge_norm = edge_norm.cuda()

        test_graph.ndata.update({'id': test_node_id,'norm': test_norm}) #'id': test_node_id,
        test_graph.edata.update({'type': test_rel, 'norm': edge_norm})

        # build adj list and calculate degrees for sampling
        adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

        # training loop
        print("start training...")

        epoch = 0
        best_f1 = 0
        while True:
            model.train()
            epoch += 1

            g, node_id, edge_type, node_norm, data, labels, edge_norm = \
                utils.generate_sampled_graph_and_labels(
                    train_data, args.graph_batch_size, args.graph_split_size,
                    num_rels, adj_list, degrees, args.negative_sample)

            node_id = torch.from_numpy(node_id).long()
            edge_type = torch.from_numpy(edge_type)
            edge_type = edge_type.long()
            node_norm = torch.from_numpy(node_norm)
            data, labels = torch.from_numpy(data), torch.from_numpy(labels)
            deg = g.in_degrees(range(g.number_of_nodes())).float()
            edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)
            if use_cuda:
                node_id, deg = node_id.cuda(), deg.cuda()
                edge_type, node_norm, edge_norm = edge_type.cuda(), node_norm.cuda(), edge_norm.cuda()
                data, labels = data.cuda(), labels.cuda()
            g.ndata.update({'id': node_id, 'norm': node_norm})
            g.edata['type'] = edge_type
            g.edata['norm'] = edge_norm

            loss = model.get_loss(g, data, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()

            optimizer.zero_grad()

            # validation
            if epoch % args.evaluate_every == 0:
                # perform validation on CPU because full graph is too large
                if use_cuda:
                    model.cpu()
                model.eval()
                scores_val, labels_val = utils.compute_score(test_graph, model, valid_data)  # predicted probability
                scores_val = scores_val.detach().numpy()

                best_thred, f1_val = utils._select_threshold(labels_val, scores_val)

                if f1_val < best_f1:
                    if epoch >= args.n_epochs:
                        break
                else:
                    best_f1 = f1_val
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'best_threshold': best_thred}, model_state_file)
                if use_cuda:
                    model.cuda()

        print("training done")

        print("\nstart testing:")
        # use best model checkpoint
        checkpoint = torch.load(model_state_file)
        if use_cuda:
            model.cpu()  # test on CPU
        model.eval()
        model.load_state_dict(checkpoint['state_dict'])
        print("Using best epoch: {}".format(checkpoint['epoch']))
        best_thred = checkpoint['best_threshold']

        scores_test, labels_test = utils.compute_score(test_graph, model, test_data)
        scores_test = scores_test.detach().numpy()
        labels_test_pred = (scores_test > best_thred) * np.ones_like(scores_test)

        test_rules, tmp1 = utils.find_rules_bt(path, i, test_data, labels_test)
        pred_rules, tmp2 = utils.find_rules_bt(path, i, test_data, labels_test_pred)

        true_rules_out += 'fold ' + str(i) + '\n'
        true_rules_out += tmp1 + '\n'

        pred_rules_out += 'fold ' + str(i) + '\n'
        pred_rules_out += tmp2 + '\n'

        precision_test, recall_test, f1_test = utils.metrics(test_rules, pred_rules)

        results_out += "Test Precision: {:.4f} | Test Recall: {:.4f} | Test F1: {:.4f} \n".format(
            precision_test, recall_test, f1_test)
        print("Test Precision: {:.4f} | Test Recall: {:.4f} | Test F1: {:.4f} \n".format(
            precision_test, recall_test, f1_test))

        results.append([precision_test, recall_test, f1_test])

    mean_p, mean_r, mean_f1 = np.mean(np.array(results), axis=0)

    results_out += "Mean values over " + str(fold) + " fold: Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}".format(
        mean_p, mean_r, mean_f1)
    print("Mean values over " + str(fold) + " fold: Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}".format(
        mean_p, mean_r, mean_f1))

    file_name = args.ftype + '_' + args.dataset + '.txt'

    f = open('result/' + file_name, 'w', encoding='utf-8')
    f.write(results_out)
    f.close()

    f = open('rules/true_' + file_name, 'w', encoding='utf-8')
    f.write(true_rules_out)
    f.close()

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
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=500,
            help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, default='wine', # required=True,
            help="dataset to use")
    parser.add_argument("-f", "--ftype", type=str, default='analogy',  # required=True,
                        help="feature type to use")
    parser.add_argument("--eval-batch-size", type=int, default=50,
            help="batch size when evaluating")
    parser.add_argument("--regularization", type=float, default=0.1,
            help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=150,
            help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
            help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=2,
            help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=10,
            help="perform evaluation every n epochs")

    args = parser.parse_args()
    print(args)
    main(args)