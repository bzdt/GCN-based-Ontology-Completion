import numpy as np
import os

def read_rules(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()

    result = dict()
    for line in lines:
        if line.startswith('\n'):
            continue
        if line.startswith('fold'):
            fid = int(line.strip().split(' ')[1])
            result[fid] = set()
        else:
            result[fid].add(line.strip())
    return result


def combine_rules(uni_file, bi_file, fold, flag=False):
    uni_rules = read_rules(uni_file)
    bi_rules = read_rules(bi_file)

    combined_rules = dict.fromkeys(range(fold), set())
    for i in range(fold):
        if i in uni_rules:
            combined_rules[i] = combined_rules[i].union(uni_rules[i])
        if i in bi_rules:
            combined_rules[i] = combined_rules[i].union(bi_rules[i])
    return combined_rules


def metrics(true_rules, pred_rules):
    result = []
    for i in range(len(true_rules)):
        if len(true_rules[i]) < 1:
            continue
        correct_num = len(true_rules[i].intersection(pred_rules[i]))
        pred_num = len(pred_rules[i])
        true_num = len(true_rules[i])
        precision = correct_num / (pred_num + 1e-10)
        recall = correct_num / (true_num + 1e-10)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-10)
        print(correct_num, pred_num, true_num)
        result.append([precision, recall, f1])
    return np.mean(np.array(result), axis=0)


def write_rules(rules, file_name):
    result = ''
    for i in range(len(rules)):
        result += 'fold ' + str(i) + '\n'
        for r in rules[i]:
            result += r + '\n'
    f = open(file_name, 'w', encoding='utf-8')
    f.write(result)
    f.close()



if __name__ == '__main__':
    datasets = ['wine']  # 'wine', 'economy', 'transport', 'olympics'
    fold = 10

    if not os.path.exists('./combined_rules'):
        os.makedirs('./combined_rules')

    result = ''
    for data in datasets:
        path = 'rules/'
        an_file_name = 'analogy_' + data + '.txt'
        em_file_name = 'embedding_' + data + '.txt'

        pred_rules = combine_rules(path + 'pred_' + an_file_name, path + 'pred_' + em_file_name, fold)
        true_rules = read_rules(path + 'true_' + an_file_name)
        for i in range(fold):
            if i not in true_rules:
                true_rules[i] = set()

        write_rules(true_rules, 'combined_rules/' + 'true_' + data + '.txt')
        write_rules(pred_rules, 'combined_rules/' + 'pred_' + data + '.txt')

        mean_p, mean_r, mean_f1 = metrics(true_rules, pred_rules)
        result += data + ': '
        result += "Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}\n".format(mean_p, mean_r, mean_f1)

    f = open('result/combined_features.txt', 'w', encoding='utf-8')
    f.write(result)
    f.close()


