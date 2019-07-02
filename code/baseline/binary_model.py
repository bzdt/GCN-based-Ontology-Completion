from sklearn.preprocessing import label_binarize
from student_t import *
from scipy.optimize import minimize_scalar

def normalize(x):
    return 1.0 / (1 + np.exp(-x))


def train(triples, node_features, pair_features_all, num_template, type, pair_dict=None, param_all=None, param_pair_all=None):
    s = triples[:, 0]
    r = triples[:, 1]
    t = triples[:, 2]

    if type == 'embedding':
        pair_features = node_features[s] - node_features[t]
        pdf_pair_all = compute_pdf_single(pair_features, param_pair_all)
    else:
        pair_idx = []
        for i, j in zip(s, t):
            pair_idx.append(pair_dict['(' + str(i) + ', ' + str(j) + ')'])
        pair_features = pair_features_all[pair_idx]
        pdf_pair_all = compute_pdf_single(pair_features, param_pair_all)

    labels = label_binarize(r, classes=np.arange(num_template))

    param_pair = train_distribution(pair_features, labels)

    pdf_pair = compute_pdf(pair_features, param_pair, num_template)

    node_features_s = node_features[s]
    param_s = train_distribution(node_features_s, labels)

    pdf_s = compute_pdf(node_features_s, param_s, num_template)
    pdf_s_all = compute_pdf_single(node_features_s, param_all)

    node_features_t = node_features[t]
    param_t = train_distribution(node_features_t, labels)

    pdf_t = compute_pdf(node_features_t, param_t, num_template)
    pdf_t_all = compute_pdf_single(node_features_t, param_all)

    prob_num = pdf_s * pdf_t * pdf_pair  # numerator
    prob_den = pdf_s_all * pdf_t_all * pdf_pair_all  # denominator

    prob = (1.0 / prob_den) * prob_num.transpose()
    prob = prob.transpose()

    lambdas = []
    for i in range(num_template):
        d = prob[:, i]
        y = labels[:, i]

        def log_likehood(lambda_t):
            p = normalize(lambda_t * d)
            non_idx = np.where(y == 0)
            p[non_idx] = 1 - p[non_idx]
            p = np.log(p + 1e-20)
            s = -np.sum(p)
            return s

        res = minimize_scalar(log_likehood, method='Golden')
        lambdas.append(res.x)

    return np.array(param_s), np.array(param_t), np.array(param_pair), np.array(lambdas)


def test(triples, node_features, pair_features_all, num_template, param_s, param_t, param_all, param_pair, param_pair_all, lambdas, type, pair_dict):
    s = triples[:, 0]
    r = triples[:, 1]
    t = triples[:, 2]

    node_features_s = node_features[s]
    node_features_t = node_features[t]

    if type == 'embedding':
        pair_features = node_features_s - node_features_t
        pdf_pair_all = compute_pdf_single(pair_features, param_pair_all)  # student-t if word embedding
    else:
        pair_idx = []
        for i, j in zip(s, t):
            if '(' + str(i) + ', ' + str(j) + ')' in pair_dict:
                pair_idx.append(pair_dict['(' + str(i) + ', ' + str(j) + ')'])
            else:
                pair_idx.append(0)
        pair_features = pair_features_all[pair_idx]
        pdf_pair_all = compute_pdf_single(pair_features, param_pair_all)

    pdf_s = compute_pdf(node_features_s, param_s, num_template)
    pdf_s_all = compute_pdf_single(node_features_s, param_all)

    pdf_t = compute_pdf(node_features_t, param_t, num_template)
    pdf_t_all = compute_pdf_single(node_features_t, param_all)

    pdf_pair = compute_pdf(pair_features, param_pair, num_template)

    prob_num = pdf_s * pdf_t * pdf_pair  # numerator
    prob_num = prob_num * lambdas.transpose()
    prob_den = pdf_s_all * pdf_t_all * pdf_pair_all  # denominator

    prob = (1.0 / prob_den) * prob_num.transpose()
    prob = prob.transpose()
    prob = 1.0 / (1.0 + np.exp(-prob))  # range to [0,1]

    true_labels = label_binarize(r, classes=np.arange(num_template))

    return prob, true_labels
