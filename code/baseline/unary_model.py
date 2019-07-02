from student_t import *
import numpy as np
from scipy.optimize import minimize_scalar


def normalize(x):
    return 1.0 / (1 + np.exp(-x))


def train(train_data, train_label, param_all):
    # compute parameters of distribution
    params_t = train_distribution(train_data, train_label)  # for each template
    num_data, num_class = train_label.shape

    pdf_t = compute_pdf(train_data, params_t, num_class)
    pdf_all = compute_pdf_single(train_data, param_all)

    t_all_l = (1.0 / pdf_all) * pdf_t.transpose()  # * train_label
    t_all_l = t_all_l.transpose()

    lambdas = []
    for i in range(num_class):
        d = t_all_l[:, i]
        y = train_label[:, i]

        def log_likehood(lambda_t):
            p = lambda_t * d
            non_idx = np.where(y == 0)
            p[non_idx] = 1 - p[non_idx]
            p = np.log(p + 1e-20)
            s = -np.sum(p)
            return s

        res = minimize_scalar(log_likehood, method='Golden')
        lambdas.append(res.x)
    return np.array(params_t), np.array(lambdas)


def test(test_data, params_t, param_all, lambdas):
    num_class = lambdas.shape[0]

    pdf_t = compute_pdf(test_data, params_t, num_class)
    pdf_all = compute_pdf_single(test_data, param_all)

    prob = lambdas.transpose() * pdf_t
    prob = (1.0 / pdf_all) * prob.transpose()
    prob = prob.transpose()
    prob = 1.0 / (1.0 + np.exp(-prob))  # range to [0,1]

    return prob
