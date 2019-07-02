import warnings
from binary_model import *
import random
from data_preprocessing_bi import train_test_triples
from utils import _select_threshold, find_rules_bt, metrics
import os

warnings.filterwarnings('ignore')
random.seed(123)

def random_sample(data, num):
    result = []
    for i in range(num):
        result.append(random.choice(data))
    return result

if __name__=='__main__':
    # load data
    dataset = ['wine']  # 'sumo','economy', 'olympics', 'transport', 'wine'
    feature_type = ['embedding', 'analogy']
    fold = 10

    if not os.path.exists('./result/binary/'):
        os.makedirs('./result/binary/')

    if not os.path.exists('./rules/binary/'):
        os.makedirs('rules/binary/')

    for data in dataset:
        for type in feature_type:
            result = []
            output = ''
            true_rules_out = ''
            pred_rules_out = ''
            out_file = type + '_' + data + '.txt'

            for i in range(fold):
                path = 'dataset/binary/' + data + '/' + str(fold) + '_fold/'
                train_data, test_data, node_features, pair_features_all, pair_dict, num_rel = train_test_triples(path, i, type)

                if test_data.shape[0] < 1:
                    continue

                s = train_data[:, 0]
                r = train_data[:, 1]
                t = train_data[:, 2]

                # param_all = train_gaussian(node_features)  # param_all: denominator for G_s and G_t
                param_all = train_single_distribution(node_features)

                if type == 'analogy':
                    pair_idx = []
                    for k,j in zip(s,t):
                        pair_idx.append(pair_dict['('+str(k)+', '+str(j)+')'])
                    pair_features = pair_features_all[pair_idx]
                    param_pair_all = train_single_distribution(pair_features)
                else:
                    rand_s = random_sample(s, 2 * len(s))
                    rand_t = random_sample(t, 2 * len(t))
                    pair_features = node_features[rand_s] - node_features[rand_t]
                    param_pair_all = train_single_distribution(pair_features)

                train_idx = np.arange(train_data.shape[0])
                valid_idx = sorted(random.sample(list(train_idx), len(train_idx) // 9))
                train_idx = sorted(list(set(train_idx) - set(valid_idx)))
                train_data0, valid_data = train_data[train_idx], train_data[valid_idx]

                param_s, param_t, param_pair, lambdas = train(train_data0, node_features, pair_features_all, num_rel, type, pair_dict, param_all, param_pair_all)

                prob_labels_val, val_labels = test(valid_data, node_features, pair_features_all, num_rel, param_s, param_t, param_all, param_pair, param_pair_all, lambdas, type, pair_dict)
                best_thred = _select_threshold(val_labels, prob_labels_val)

                prob_labels, test_labels = test(test_data, node_features, pair_features_all, num_rel, param_s, param_t, param_all, param_pair, param_pair_all, lambdas, type, pair_dict)
                pred_labels = np.zeros(prob_labels.shape)

                pred_labels[np.where(prob_labels > best_thred)] = 1

                test_rules, tmp1 = find_rules_bt(path, i, test_data, test_labels)
                pred_rules, tmp2 = find_rules_bt(path, i, test_data, pred_labels)

                true_rules_out += 'fold ' + str(i) + '\n'
                true_rules_out += tmp1 + '\n'

                pred_rules_out += 'fold ' + str(i) + '\n'
                pred_rules_out += tmp2 + '\n'

                precision_test, recall_test, f1_test = metrics(test_rules, pred_rules)

                output += 'fold ' + str(i) + ': '
                output += 'Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f} \n'.format(precision_test, recall_test, f1_test)

                result.append([precision_test, recall_test, f1_test])
                break

            mean_p, mean_r, mean_f1 = np.mean(np.array(result), axis=0)
            output += "Mean values over" + str(
                    fold) + " : Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}\n".format(mean_p, mean_r, mean_f1)

            f = open('result/binary/' + out_file, 'w', encoding='utf-8')
            f.writelines(output)
            f.close()

            f = open('rules/binary/' + 'true_' + out_file, 'w', encoding='utf-8')
            f.writelines(true_rules_out)
            f.close()

            f = open('rules/binary/' + 'pred_' + out_file, 'w', encoding='utf-8')
            f.writelines(pred_rules_out)
            f.close()

