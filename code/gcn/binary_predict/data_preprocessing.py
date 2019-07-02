from sklearn.decomposition import PCA
from word_embedding import word_embedding
from utils import _read_dictionary, _read_triplets
import os
import numpy as np
import csv
import pandas as pd


def train_test_triples(path, fold, ftype, dim_num):
    train_path = path + 'train/s' + str(fold + 1)
    test_path = path + 'test/s' + str(fold + 1)
    node_dict_file = train_path + '/nodes.dict'
    rel_dict_file = train_path + '/binary_templates.dict'
    uni_train_file = train_path + '/binary_template_triples_uni.txt'
    bi_train_file = train_path + '/binary_template_triples_bi.txt'
    train_data = []

    if os.path.exists(node_dict_file) and os.path.exists(rel_dict_file):
        nodes_dict = _read_dictionary(node_dict_file)
        rel_dict = _read_dictionary(rel_dict_file)
    else:
        # build node dict and rel dict
        nodes_dict = dict()
        rel_dict = dict()
        rel_id = 0
        n_id = 0
        nodes_dict_str = ''
        rel_dict_str = ''
        for triple in _read_triplets(uni_train_file):
            if triple[0] not in nodes_dict:
                nodes_dict[triple[0]] = n_id
                nodes_dict_str += str(n_id) + '\t' + triple[0] + '\n'
                n_id += 1
            if triple[2] not in nodes_dict:
                nodes_dict[triple[2]] = n_id
                nodes_dict_str += str(n_id) + '\t' + triple[2] + '\n'
                n_id += 1
            if triple[1] not in rel_dict:
                rel_dict[triple[1]] = rel_id
                rel_dict_str += str(rel_id) + '\t' + triple[1] + '\n'
                rel_id += 1
        for triple in _read_triplets(bi_train_file):
            if triple[0] not in nodes_dict:
                nodes_dict[triple[0]] = n_id
                nodes_dict_str += str(n_id) + '\t' + triple[0] + '\n'
                n_id += 1
            if triple[2] not in nodes_dict:
                nodes_dict[triple[2]] = n_id
                nodes_dict_str += str(n_id) + '\t' + triple[2] + '\n'
                n_id += 1
            if triple[1] not in rel_dict:
                rel_dict[triple[1]] = rel_id
                rel_dict_str += str(rel_id) + '\t' + triple[1] + '\n'
                rel_id += 1
        f = open(node_dict_file, 'w', encoding='utf-8')
        f.write(nodes_dict_str)
        f.close()

        f = open(rel_dict_file, 'w', encoding='utf-8')
        f.write(rel_dict_str)
        f.close()

    print("Nodes: ", len(nodes_dict))
    print("Relations: ", len(rel_dict))
    # uni-edges
    for triple in _read_triplets(uni_train_file):
        s = nodes_dict[triple[0]]
        r = rel_dict[triple[1]]
        t = nodes_dict[triple[2]]
        train_data.append([s, r, t])

    # bi-edges
    for triple in _read_triplets(bi_train_file):
        s = nodes_dict[triple[0]]
        r = rel_dict[triple[1]]
        t = nodes_dict[triple[2]]
        train_data.append([s, r, t])
        train_data.append([t, r, s])

    if ftype == 'embedding':
        feature_file = train_path + '/em_features.csv'
    elif ftype == 'analogy':
        feature_file = train_path + '/an_features_' + str(dim_num) + '.csv'

    if not os.path.exists(feature_file):
        if ftype == 'embedding':
            embeding_file = 'dataset/GoogleNews-vectors-negative300.bin.gz'
            node_features = word_embedding(embeding_file, nodes_dict)
        elif ftype == 'analogy':
            train_nodes_label = pd.read_csv(train_path + '/predicate_label.tsv', sep='\t', encoding='utf-8')
            label_dict = dict()
            l_id = 0
            for nod, lab in zip(train_nodes_label['nodes'].values, train_nodes_label['label'].values):
                if nod not in nodes_dict:
                    continue
                lab = lab.replace(',TempateExpression', '\tTempateExpression').split('\t')
                if len(lab) > 1:
                    for l in lab:
                        if l not in label_dict:
                            label_dict[l] = l_id
                            l_id = l_id + 1
                else:
                    if lab[0] not in label_dict:
                        label_dict[lab[0]] = l_id
                        l_id = l_id + 1

            matrix = np.zeros((len(nodes_dict), len(label_dict)))
            for nod, lab in zip(train_nodes_label['nodes'].values, train_nodes_label['label'].values):
                if nod not in nodes_dict:
                    continue
                lab = lab.replace(',TempateExpression', '\tTempateExpression').split('\t')
                if len(lab) > 1:
                    for l in lab:
                        matrix[nodes_dict[nod], label_dict[l]] = 1
                else:
                    matrix[nodes_dict[nod], label_dict[lab[0]]] = 1

            pca = PCA(n_components=dim_num)
            node_features = pca.fit_transform(matrix)
        f = open(feature_file, 'w', encoding='utf-8', newline='')
        writer = csv.writer(f)
        writer.writerow(node_features[0])
        writer.writerows(node_features)
        f.close()
    else:
        node_features = pd.read_csv(feature_file, sep=',', encoding='utf-8')

    uni_test_file = test_path + '/binary_template_triples_uni.txt'
    train_data = np.array(train_data)
    train_nodes = set(train_data[:, 0]).union(set(train_data[:, 2]))
    test_data = []
    for triple in _read_triplets(uni_test_file):
        if triple[0] not in nodes_dict:
            continue
        if triple[2] not in nodes_dict:
            continue
        if triple[1] not in rel_dict:
            continue
        s = nodes_dict[triple[0]]
        r = rel_dict[triple[1]]
        t = nodes_dict[triple[2]]
        if s in train_nodes and t in train_nodes:
            test_data.append([s, r, t])

    bi_test_file = test_path + '/binary_template_triples_bi.txt'
    for triple in _read_triplets(bi_test_file):
        if triple[0] not in nodes_dict:
            continue
        if triple[2] not in nodes_dict:
            continue
        if triple[1] not in rel_dict:
            continue
        s = nodes_dict[triple[0]]
        r = rel_dict[triple[1]]
        t = nodes_dict[triple[2]]
        if s in train_nodes and t in train_nodes:
            test_data.append([s, r, t])
            test_data.append([t, r, s])

    return train_data, np.array(test_data), len(nodes_dict), len(rel_dict), node_features


def load_whole_data(path, ftype, dim_num):
    train_path = path + 'train/s1'
    test_path = path + 'test/s1'
    node_dict_file = train_path + '/all_nodes.dict'
    rel_dict_file = train_path + '/all_binary_templates.dict'
    uni_file = '/binary_template_triples_uni.txt'
    bi_file = '/binary_template_triples_bi.txt'
    data = []

    if os.path.exists(node_dict_file) and os.path.exists(rel_dict_file):
        nodes_dict = _read_dictionary(node_dict_file)
        rel_dict = _read_dictionary(rel_dict_file)
    else:
        # build node dict and rel dict
        nodes_dict = dict()
        rel_dict = dict()
        rel_id = 0
        n_id = 0
        nodes_dict_str = ''
        rel_dict_str = ''
        # train data
        for triple in _read_triplets(train_path + uni_file):
            if triple[0] not in nodes_dict:
                nodes_dict[triple[0]] = n_id
                nodes_dict_str += str(n_id) + '\t' + triple[0] + '\n'
                n_id += 1
            if triple[2] not in nodes_dict:
                nodes_dict[triple[2]] = n_id
                nodes_dict_str += str(n_id) + '\t' + triple[2] + '\n'
                n_id += 1
            if triple[1] not in rel_dict:
                rel_dict[triple[1]] = rel_id
                rel_dict_str += str(rel_id) + '\t' + triple[1] + '\n'
                rel_id += 1
        for triple in _read_triplets(train_path + bi_file):
            if triple[0] not in nodes_dict:
                nodes_dict[triple[0]] = n_id
                nodes_dict_str += str(n_id) + '\t' + triple[0] + '\n'
                n_id += 1
            if triple[2] not in nodes_dict:
                nodes_dict[triple[2]] = n_id
                nodes_dict_str += str(n_id) + '\t' + triple[2] + '\n'
                n_id += 1
            if triple[1] not in rel_dict:
                rel_dict[triple[1]] = rel_id
                rel_dict_str += str(rel_id) + '\t' + triple[1] + '\n'
                rel_id += 1

        # test data
        for triple in _read_triplets(test_path + uni_file):
            if triple[0] not in nodes_dict:
                nodes_dict[triple[0]] = n_id
                nodes_dict_str += str(n_id) + '\t' + triple[0] + '\n'
                n_id += 1
            if triple[2] not in nodes_dict:
                nodes_dict[triple[2]] = n_id
                nodes_dict_str += str(n_id) + '\t' + triple[2] + '\n'
                n_id += 1
            if triple[1] not in rel_dict:
                rel_dict[triple[1]] = rel_id
                rel_dict_str += str(rel_id) + '\t' + triple[1] + '\n'
                rel_id += 1
        for triple in _read_triplets(test_path + uni_file):
            if triple[0] not in nodes_dict:
                nodes_dict[triple[0]] = n_id
                nodes_dict_str += str(n_id) + '\t' + triple[0] + '\n'
                n_id += 1
            if triple[2] not in nodes_dict:
                nodes_dict[triple[2]] = n_id
                nodes_dict_str += str(n_id) + '\t' + triple[2] + '\n'
                n_id += 1
            if triple[1] not in rel_dict:
                rel_dict[triple[1]] = rel_id
                rel_dict_str += str(rel_id) + '\t' + triple[1] + '\n'
                rel_id += 1

        f = open(node_dict_file, 'w', encoding='utf-8')
        f.write(nodes_dict_str)
        f.close()

        f = open(rel_dict_file, 'w', encoding='utf-8')
        f.write(rel_dict_str)
        f.close()

    print("Nodes: ", len(nodes_dict))
    print("Relations: ", len(rel_dict))

    ##===train data===========
    # uni-edges
    for triple in _read_triplets(train_path + uni_file):
        s = nodes_dict[triple[0]]
        r = rel_dict[triple[1]]
        t = nodes_dict[triple[2]]
        data.append([s, r, t])

    # bi-edges
    for triple in _read_triplets(train_path + bi_file):
        s = nodes_dict[triple[0]]
        r = rel_dict[triple[1]]
        t = nodes_dict[triple[2]]
        data.append([s, r, t])
        # train_data.append([t, r, s])

    ##===test data===========
    # uni-edges
    for triple in _read_triplets(test_path + uni_file):
        s = nodes_dict[triple[0]]
        r = rel_dict[triple[1]]
        t = nodes_dict[triple[2]]
        data.append([s, r, t])

    # bi-edges
    for triple in _read_triplets(test_path + bi_file):
        s = nodes_dict[triple[0]]
        r = rel_dict[triple[1]]
        t = nodes_dict[triple[2]]
        data.append([s, r, t])
        # train_data.append([t, r, s])

    if ftype == 'embedding':
        feature_file = train_path + '/all_em_features.csv'
    elif ftype == 'analogy':
        feature_file = train_path + '/all_an_features_' + str(dim_num) + '.csv'

    if not os.path.exists(feature_file):
        if ftype == 'embedding':
            embeding_file = 'dataset/GoogleNews-vectors-negative300.bin.gz'
            node_features = word_embedding(embeding_file, nodes_dict)
        elif ftype == 'analogy':
            train_nodes_label = pd.read_csv(train_path + '/predicate_label.tsv', sep='\t', encoding='utf-8')
            test_nodes_label = pd.read_csv(test_path + '/predicate_label.tsv', sep='\t', encoding='utf-8')
            nodes_label = pd.concat([train_nodes_label, test_nodes_label])
            label_dict = dict()
            l_id = 0
            for nod, lab in zip(nodes_label['nodes'].values, nodes_label['label'].values):
                if nod not in nodes_dict:
                    continue
                lab = lab.replace(',TempateExpression', '\tTempateExpression').split('\t')
                if len(lab) > 1:
                    for l in lab:
                        if l not in label_dict:
                            label_dict[l] = l_id
                            l_id = l_id + 1
                else:
                    if lab[0] not in label_dict:
                        label_dict[lab[0]] = l_id
                        l_id = l_id + 1

            matrix = np.zeros((len(nodes_dict), len(label_dict)))
            for nod, lab in zip(nodes_label['nodes'].values, nodes_label['label'].values):
                if nod not in nodes_dict:
                    continue
                lab = lab.replace(',TempateExpression', '\tTempateExpression').split('\t')
                if len(lab) > 1:
                    for l in lab:
                        matrix[nodes_dict[nod], label_dict[l]] = 1
                else:
                    matrix[nodes_dict[nod], label_dict[lab[0]]] = 1

            pca = PCA(n_components=dim_num)
            node_features = pca.fit_transform(matrix)

        f = open(feature_file, 'w', encoding='utf-8', newline='')
        writer = csv.writer(f)
        writer.writerow(node_features[0])
        writer.writerows(node_features)
        f.close()
    else:
        node_features = pd.read_csv(feature_file, sep=',', encoding='utf-8')

    return np.array(data), len(nodes_dict), len(rel_dict), node_features
