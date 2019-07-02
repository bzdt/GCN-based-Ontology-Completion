"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""
import numpy as np
import torch
import dgl
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from collections import defaultdict

#######################################################################
#
# Utility function for finding predicted rules
#
#######################################################################
## find rules for unary templates
# dataset: wine, sumo, ...
# template_ids: unary template ids
# pred: True if templates are predicted, False if templates are groudtruth
def find_rules_ut(path, fold, node_ids, template_ids):
    node_dict_file = path + 'train/s' + str(fold + 1) + '/nodes.dict'
    template_dict_file = path + 'train/s' + str(fold + 1) + '/unary_templates.dict'
    rules = []
    result = ''

    node_dict = read_dictionary(node_dict_file)  # {id: name}
    template_dict = read_dictionary(template_dict_file)  # {id: template name}

    for idx in range(len(node_ids)):
        n_id = node_ids[idx]
        n_name = node_dict[n_id]
        tmp_ids = np.where(template_ids[idx] > 0)[0]
        if not tmp_ids.size:
            continue
        for id in tmp_ids:
            tmp_name = template_dict[id]
            rule = tmp_name.replace('TempateExpression', 'ER')
            rule = rule.replace('?', n_name)
            rules.append(rule)
            result += rule + '\n'
    print(result)
    return rules, result

## find rules for binary templates
# dataset: wine, economy,...
# triples: test triples with id, i.e. (n_id, bt_id, n_id)
# pred_templates: predicated bt_ids, note: there may be more than one bt id for each node pair
def find_rules_bt(path, fold, triples, pred_templates):
    node_dict_file = path + 'train/s' + str(fold + 1) + '/nodes.dict'
    template_dict_file = path + 'train/s' + str(fold + 1) + '/binary_templates.dict'

    result = ''
    rules = []

    node_dict = read_dictionary(node_dict_file)  # {id: name}
    template_dict = read_dictionary(template_dict_file)  # {id: template name}
    s = triples[:,0]
    r = triples[:,1]
    t = triples[:,2]
    for idx in range(len(s)):
        s_id = s[idx]
        t_id = t[idx]
        s_name = node_dict[int(s_id)]
        t_name = node_dict[int(t_id)]
        tmp_ids = np.where(pred_templates[idx] > 0)[0]
        if not tmp_ids.size:
            continue
        for tid in tmp_ids:
            if tid not in r:
                continue
            tmp_name = template_dict[tid].replace('TempateExpression', 'ER')
            tmp_name = tmp_name.replace('?', s_name, 1)
            tmp_name = tmp_name.replace('?', t_name, 1)
            rules.append(tmp_name)
            result += tmp_name + '\n'
    return rules, result


## find top k predicted rules that not in the train set
def find_rules_bt_all(path, test_triples, test_label, test_score, topk):
    node_dict_file = path + 'train/s1' + '/all_nodes.dict'
    template_dict_file = path + 'train/s1' + '/all_binary_templates.dict'
    result = ''

    node_dict = read_dictionary(node_dict_file)  # {id: name}
    template_dict = read_dictionary(template_dict_file)  # {id: template name}

    test_score[np.where(test_label == 1)] = 0  # ignore the true rules in train set

    topk_idx = test_score.reshape(-1).argsort()[::-1][0:topk]

    s = test_triples[:, 0]
    r = test_triples[:, 1]
    t = test_triples[:, 2]
    for idx in topk_idx:
        row_idx = int(idx // test_score.shape[1])
        t_idx = int(idx % test_score.shape[1])
        if test_score[row_idx, t_idx] <= 0:
            break
        s_name = node_dict[int(s[row_idx])]
        t_name = node_dict[int(t[row_idx])]
        tmp_name = template_dict[t_idx].replace('TempateExpression', 'ER')
        tmp_name = tmp_name.replace('?', s_name, 1)
        tmp_name = tmp_name.replace('?', t_name, 1)
        result += tmp_name + '\n'

    return result

#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################
def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):   # i can be regarded as edge_id
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list]) # out degree
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    """ Edge neighborhood sampling to reduce training graph size
    """
    edges = np.zeros(sample_size, dtype=np.int32)

    #initialize
    sample_counts = np.array([d for d in degrees])  # sample count for each node
    picked = np.array([False for _ in range(n_triplets)])   # num_triple * 1
    seen = np.array([False for _ in degrees])  # num_node * 1
    for i in range(0, sample_size):
        weights = sample_counts * (~seen)
        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0
        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        #print(chosen_vertex)
        while len(adj_list[chosen_vertex]) == 0:
            chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                             p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        if sample_counts[chosen_vertex] < 1:
            seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]   # can be regarded as edge_id

        while picked[edge_number]:  # if edge has been choosed before, then choose again
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        if sample_counts[other_vertex] < 1:
            seen[other_vertex] = True
    return edges


def generate_sampled_graph_and_labels(triplets, sample_size, split_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    edges = sample_edge_neighborhood(adj_list, degrees, len(triplets),
                                     sample_size)

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
    g, rel, norm, edge_norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                             (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels, edge_norm


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm


def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph.
        some edges are binary, but others single
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph(multigraph=False)
    g.add_nodes(num_nodes)

    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)

    # src, rel, dst = triplets
    # g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    # print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))

    # normalize by dst degree, compute degrees according to edge_type
    _, inverse_index, count = np.unique((dst, rel), axis=1, return_inverse=True,
                                        return_counts=True)
    degrees = count[inverse_index]  # c_{i,r} for each relation type
    edge_norm = np.ones(len(dst), dtype=np.float32) / degrees.astype(np.float32)
    return g, rel, norm, edge_norm


def generate_graph_with_negative(triplets, num_rels, negative_rate):
    src, rel, dst = triplets.transpose()
    uniq_v, _ = np.unique((src, dst), return_inverse=True)
    # negative sampling
    samples, labels = negative_sampling(triplets, len(uniq_v),
                                     negative_rate)

    # src, rel, dst = samples.transpose()
    # build DGL graph
    g, rel, norm, edge_norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                                        (src, rel, dst))

    return g, uniq_v, rel, norm, samples, labels, edge_norm


def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    # print("Test graph:")
    return build_graph_from_triplets(num_nodes, num_rels, (src, rel, dst))

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate) # node_id
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

#######################################################################
#
# Utility function for evaluations
#
#######################################################################
# TODO: implement the prediction
# return (micro-)precision, recall, F1, mAP
def compute_score(test_graph, model, test_triplets):
    with torch.no_grad():
        embedding, w = model.evaluate(test_graph)
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        t = test_triplets[:, 2]

        emb_s = embedding[s]  # num_node * num_fea
        emb_t = embedding[t]

        emb_s = emb_s.transpose(0, 1).unsqueeze(2)  # num_fea * num_node * 1
        w = w.transpose(0, 1).unsqueeze(1)  # num_fea * 1 * num_rel
        mult_sr = torch.bmm(emb_s, w)  # emb_s * w, size = num_fea * num_node * num_rel

        mult_sr = mult_sr.transpose(1, 2)  # num_fea * num_rel * num_node
        mult_sr = mult_sr.transpose(0, 2)  # num_node * num_rel * num_fea
        emb_t = emb_t.unsqueeze(2)  # num_node * num_fea * 1

        products = torch.bmm(mult_sr, emb_t)  # num_node * num_rel * 1
        products = products.squeeze(2)  # num_node * num_rel
        score = torch.sigmoid(products)  # num_node * num_rel

        
        y_true = label_binarize(r, classes=np.arange(w.size()[2]))  # num_node * num_relss
    return score, y_true 

# select best threshold
def select_threshold(y_true, y_prob, num_class):
    best_thred = []
    for i in range(num_class):
        thresholds = sorted(set(y_prob[:,i]))
        f1 = []
        for th in thresholds:
            y_pred = (y_prob[:,i] > th) * np.ones_like(y_prob[:,i])
            f1.append(f1_score(y_true[:,i], y_pred, average='micro'))
        best_thred.append(thresholds[int(np.argmax(np.array(f1)))])
    return best_thred


def _select_threshold(y_true, y_prob):
    thresholds = sorted(set(y_prob.reshape(-1)))
    f1 = []
    for th in thresholds:
        y_pred = (y_prob> th) * np.ones_like(y_prob)
        f1.append(f1_score(y_true, y_pred, average='micro'))
    best_thred = thresholds[int(np.argmax(np.array(f1)))]
    return best_thred, np.max(np.array(f1))


def metrics(test_rules, pred_rules):
    acc_num = len(set(test_rules).intersection(set(pred_rules)))
    precision = acc_num / (len(set(pred_rules)) + 1e-10)
    recall = acc_num / (len(set(test_rules)) + 1e-10)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1


# return tuple like (etype, list of index)
def find_same_etype(etype):
    tally = defaultdict(list)
    for i, item in enumerate(etype):
        tally[int(item)].append(i)

    return ((key, locs) for key, locs in tally.items()
            if len(locs) > 1)



# key = template name, value = id
def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d

# key = id, value = template name
def read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[int(line[0])] = line[1]
    return d

def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line

def _read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in _read_triplets(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l