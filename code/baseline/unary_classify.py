from unary_model import *
import warnings
from student_t import *
import random
from data_preprocessing_uni import train_test_idx
from utils import _select_threshold, select_threshold, find_rules_ut, metrics
import os
warnings.filterwarnings('ignore')
random.seed(123)

if __name__ == '__main__':
    # load data
    dataset = ['wine']  # 'economy', 'olympics', 'transport', 'wine', 'sumo
    feature_type = ['embedding', 'analogy']  # analogy, embedding
    fold = 10

    if not os.path.exists('./result/unary/'):
        os.makedirs('./result/unary/')

    if not os.path.exists('./rules/unary/'):
        os.makedirs('rules/unary/')

    for type in feature_type:
        for data in dataset:
            result = []
            output = ''
            true_rules_out = ''
            pred_rules_out = ''
            out_file = type + '_' + data + '.txt'

            for i in range(fold):
                path = 'dataset/unary/' + data + '/' + str(fold) + '_fold/'
                features, train_labels, test_labels, train_idx, test_idx = train_test_idx(path, i, type)
                tmp_label = train_labels


                test_template_id = np.where(test_labels.sum(axis=0) > 0)[0]

                if test_labels.shape[0] == 0:
                    continue

                num_class = train_labels.shape[1]

                param_all = train_single_distribution(features)

                valid_idx = sorted(random.sample(list(train_idx), len(train_idx) // 9))
                train_idx = sorted(list(set(train_idx) - set(valid_idx)))

                train_data, valid_data, test_data = features[train_idx], features[valid_idx], features[test_idx]
                train_labels, valid_labels = train_labels[train_idx], train_labels[valid_idx]

                params_t, lambdas = train(train_data, train_labels, param_all)

                prob_val = test(valid_data, params_t, param_all, lambdas)
                best_thred = _select_threshold(valid_labels, prob_val)

                prob_labels = test(test_data, params_t, param_all, lambdas)

                pred_labels = np.zeros(prob_labels.shape)
                pred_labels[np.where(prob_labels > best_thred)] = 1

                test_rules, str_terule = find_rules_ut(path, i, test_idx, tmp_label, test_labels, test_template_id)
                pred_rules, str_prerule = find_rules_ut(path, i, test_idx, tmp_label, pred_labels, test_template_id, pred=True)

                true_rules_out += 'fold ' + str(i) + '\n'
                true_rules_out += str_terule
                pred_rules_out += 'fold ' + str(i) + '\n'
                pred_rules_out += str_prerule

                test_precision, test_recall, test_f1 = metrics(test_rules, pred_rules)

                output += 'fold ' + str(i) + ': '
                output += 'Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f} \n'.format(test_precision, test_recall, test_f1)
                result.append([test_precision, test_recall, test_f1])
                break

            mean_p, mean_r, mean_f1 = np.mean(np.array(result), axis=0)
            output += "Mean values over" + str(fold) + " : Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}\n".format(mean_p, mean_r, mean_f1)

            f = open('result/unary/' + out_file, 'w', encoding='utf-8')
            f.writelines(output)
            f.close()

            f = open('rules/unary/' + 'true_' + out_file, 'w', encoding='utf-8')
            f.writelines(true_rules_out)
            f.close()

            f = open('rules/unary/' + 'pred_' + out_file, 'w', encoding='utf-8')
            f.writelines(pred_rules_out)
            f.close()







