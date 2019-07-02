import numpy as np
from sklearn.metrics import f1_score


def select_threshold(y_true, y_prob, num_class):
    best_thred = []
    for i in range(num_class):
        thresholds = sorted(set(y_prob[:, i]))
        f1 = []
        for th in thresholds:
            y_pred = (y_prob[:, i] > th) * np.ones_like(y_prob[:, i])
            f1.append(f1_score(y_true[:, i], y_pred, average='micro'))
        best_thred.append(thresholds[int(np.argmax(np.array(f1)))])
    return best_thred


def _select_threshold(y_true, y_prob):
    thresholds = sorted(set(y_prob.reshape((-1))))
    f1 = []
    for th in thresholds:
        y_pred = (y_prob > th) * np.ones_like(y_prob)
        f1.append(f1_score(y_true, y_pred, average='micro'))
    best_thred = thresholds[int(np.argmax(np.array(f1)))]
    return best_thred


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


def metrics(test_rules, pred_rules):
    acc_num = len(set(test_rules).intersection(set(pred_rules)))

    precision = acc_num / (len(set(pred_rules)) + 1e-10)
    recall = acc_num / (len(set(test_rules)) + 1e-10)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1


# find rules for unary templates
# template_ids: unary template ids
def find_rules_ut(path, fold, node_ids, train_label, template_ids, interested_template_ids, pred=False):
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
            t_id = id
            if train_label[n_id, t_id] == 1:
                continue
            if pred and t_id not in interested_template_ids:
                continue
            tmp_name = template_dict[int(t_id)]
            rule = tmp_name.replace('TempateExpression', 'ER')
            rule = rule.replace('?', n_name)
            rules.append(rule)
            result += rule + '\n'

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