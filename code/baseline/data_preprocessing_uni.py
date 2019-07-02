import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from word_embedding import word_embedding
import os
import csv
from utils import _read_dictionary

def train_test_idx(path, fold, ftype):
    train_path = path + 'train/s' + str(fold + 1)
    train_nodes_label = pd.read_csv(train_path + '/predicate_label.tsv', sep='\t', encoding='utf-8')
    node_dict_file = train_path + '/nodes.dict'

    if os.path.exists(node_dict_file):
        nodes_dict = _read_dictionary(node_dict_file)
    else:
        # node dict
        nodes_dict = dict()
        nodes = train_nodes_label['nodes'].values.tolist()
        nod_id = 0
        node_id = ''
        for n in nodes:
            if n not in nodes_dict:
                nodes_dict[n] = nod_id
                node_id += str(nod_id) + '\t' + n + '\n'
                nod_id += 1
        f = open(node_dict_file, 'w', encoding='utf-8')
        f.write(node_id)
        f.close()

    print('Number of training nodes: ', len(nodes_dict))

    node_id_con = set()
    relation_dict = dict()
    rid = 1  # 0 for self-relation, last id + 1 for relation to node "top"
    edge_list = []

    # uni-edges
    f = open(train_path + '/binary_template_triples_uni.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()

    for line in lines:
        triple = line.strip().split('\t')
        if triple[0] in nodes_dict and triple[2] in nodes_dict:
            if triple[1] not in relation_dict:
                relation_dict[triple[1]] = rid
                rid = rid + 1
            src = nodes_dict[triple[0]]
            dst = nodes_dict[triple[2]]
            node_id_con.add(src)
            node_id_con.add(dst)
            edge_list.append((src, dst, relation_dict[triple[1]]))

    # bi-edges
    f = open(train_path + '/binary_template_triples_bi.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()

    for line in lines:
        triple = line.strip().split('\t')
        if triple[0] in node_id_con and triple[2] in node_id_con:
            if triple[1] not in relation_dict:
                relation_dict[triple[1]] = rid
                rid = rid + 1
            src = nodes_dict[triple[0]]
            dst = nodes_dict[triple[2]]
            node_id_con.add(src)
            node_id_con.add(dst)
            edge_list.append((src, dst, relation_dict[triple[1]]))
            edge_list.append((dst, src, relation_dict[triple[1]]))

    for n in node_id_con:  # the graph only includes nodes with edges
        edge_list.append((n, n, 0))

    edge_list = sorted(edge_list, key=lambda x: (x[1], x[0], x[2]))
    edge_list = np.array(edge_list, dtype=np.int)

    # unary templates (node labels)
    label_dict_file = train_path + '/unary_templates.dict'
    if os.path.exists(label_dict_file):
        label_dict = _read_dictionary(label_dict_file)
    else:
        label_dict = dict()
        l_id = 0
        label_id = ''
        for nod, lab in zip(train_nodes_label['nodes'].values, train_nodes_label['label'].values):
            if nod not in nodes_dict:
                continue
            lab = lab.replace(',TempateExpression', '\tTempateExpression').split('\t')
            if len(lab) > 1:
                for l in lab:
                    if l not in label_dict:
                        label_dict[l] = l_id
                        label_id += str(l_id) + '\t' + l + '\n'
                        l_id = l_id + 1
            else:
                if lab[0] not in label_dict:
                    label_dict[lab[0]] = l_id
                    label_id += str(l_id) + '\t' + lab[0] + '\n'
                    l_id = l_id + 1
            f = open(label_dict_file, 'w', encoding='utf-8')
            f.write(label_id)
            f.close()

    num_node = len(nodes_dict)
    num_rel = len(relation_dict) + 1
    train_labels = sp.lil_matrix((num_node, len(label_dict)))
    for nod, lab in zip(train_nodes_label['nodes'].values, train_nodes_label['label'].values):
        if nod not in nodes_dict:
            continue
        nod_id = nodes_dict[nod]
        lab = lab.replace(',TempateExpression', '\tTempateExpression').split('\t')
        if len(lab) > 1:
            for l in lab:
                lab_id = label_dict[l]
                train_labels[nod_id, lab_id] = 1
        else:
            lab_id = label_dict[lab[0]]
            train_labels[nod_id, lab_id] = 1

    train_labels = train_labels.tocsr().todense()

    test_path = path + 'test/s' + str(fold + 1)
    test_nodes_label = pd.read_csv(test_path + '/predicate_label.tsv', sep='\t', encoding='utf-8')
    test_nodes = test_nodes_label['nodes']
    test_idx = []
    for n in test_nodes:
        if n in nodes_dict and nodes_dict[n] in node_id_con:
            test_idx.append(nodes_dict[n])
    test_idx = sorted(list(set(test_idx)))

    test_labels = sp.lil_matrix((len(test_idx), len(label_dict)))
    i = 0
    for nod, lab in zip(test_nodes_label['nodes'].values, test_nodes_label['label'].values):
        if nod not in nodes_dict:
            i += 1
            continue
        n_id = nodes_dict[nod]
        if n_id not in test_idx:
            i += 1
            continue
        j = test_idx.index(n_id)
        lab = lab.replace(',TempateExpression', '\tTempateExpression').split('\t')
        if len(lab) > 1:
            for l in lab:
                if l in label_dict:
                    lab_id = label_dict[l]
                    test_labels[j, lab_id] = 1
        else:
            if lab[0] in label_dict:
                lab_id = label_dict[lab[0]]
                test_labels[j, lab_id] = 1
    print(i, 'test nodes not in training set')

    test_labels = test_labels.tocsr()

    ## remove nodes that all the labels are 0
    idx = np.where(test_labels.toarray().sum(axis=1) > 0)[0]
    test_labels = test_labels[idx]
    test_idx = np.array(test_idx)[idx]

    test_labels = test_labels.tocsr()

    if ftype == 'embedding':
        feature_file = train_path + '/em_features.csv'
    elif ftype == 'analogy':
        feature_file = train_path + '/an_features.csv'

    if not os.path.exists(feature_file):
        if ftype == 'embedding':
            embedding_file = 'dataset/GoogleNews-vectors-negative300.bin.gz'
            node_features = word_embedding(embedding_file, nodes_dict)
        elif ftype == 'analogy':
            pca = PCA(n_components=40)
            node_features = pca.fit_transform(train_labels)
        f = open(feature_file, 'w', encoding='utf-8', newline='')
        writer = csv.writer(f)
        writer.writerow(node_features[0])
        writer.writerows(node_features)
        f.close()
    else:
        node_features = pd.read_csv(feature_file, sep=',', encoding='utf-8')

    return np.array(node_features), np.array(train_labels), test_labels.toarray(), node_id_con, test_idx

