from sklearn.decomposition import PCA
from word_embedding import word_embedding
from utils import _read_dictionary, _read_triplets, read_dictionary
import os
import numpy as np
import csv
import pandas as pd


def train_test_triples(path, fold, ftype):
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
        feature_file = train_path + '/an_features.csv'

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

            pca = PCA(n_components=40)
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

    if ftype == 'analogy':
        rel_features_file = train_path + '/pair_features.csv'
        pair_file = train_path + '/pair_dict.txt'
        if not os.path.exists(rel_features_file) and not os.path.exists(pair_file):
            pair_dict = dict()
            pair_id = 0
            for triple in _read_triplets(uni_train_file):
                src = triple[0]
                dst = triple[2]
                pair = nodes_dict[src], nodes_dict[dst]
                if str(pair) not in pair_dict:
                    pair_dict[str(pair)] = pair_id
                    pair_id += 1

            for triple in _read_triplets(bi_train_file):
                src = triple[0]
                dst = triple[2]
                pair1 = nodes_dict[src], nodes_dict[dst]
                pair2 = nodes_dict[dst], nodes_dict[src]
                if pair1 not in pair_dict:
                    pair_dict[str(pair1)] = pair_id
                    pair_id += 1
                if pair2 not in pair_dict:
                    pair_dict[str(pair2)] = pair_id
                    pair_id += 1

            num_rel = len(rel_dict)
            num_pair = len(pair_dict)

            matrix = np.zeros((num_rel, num_pair))
            for triple in _read_triplets(uni_train_file):
                src = nodes_dict[triple[0]]
                rel = rel_dict[triple[1]]
                dst = nodes_dict[triple[2]]
                key = src, dst
                matrix[rel, pair_dict[str(key)]] = 1

            for triple in _read_triplets(bi_train_file):
                src = nodes_dict[triple[0]]
                rel = rel_dict[triple[1]]
                dst = nodes_dict[triple[2]]
                key1 = src, dst
                key2 = dst, src
                matrix[rel, pair_dict[str(key1)]] = 1
                matrix[rel, pair_dict[str(key2)]] = 1

            f = open(rel_features_file, 'w', encoding='utf-8', newline='')
            writer = csv.writer(f)
            writer.writerow(matrix[0])
            writer.writerows(matrix)
            f.close()

            out = ''
            for key in pair_dict:
                out += str(pair_dict[key]) + '\t' + str(key) + '\n'

            f = open(pair_file, 'w', encoding='utf-8')
            f.writelines(out)
            f.close()
        else:
            matrix = pd.read_csv(rel_features_file)
            pair_dict = _read_dictionary(pair_file)

        pca = PCA(n_components=32)
        pair_features = pca.fit_transform(matrix.transpose())  # pair_num * 32
    else:
        pair_features = None
        pair_dict = None

    return np.array(train_data), np.array(test_data), np.array(node_features), np.array(pair_features), pair_dict, len(rel_dict)