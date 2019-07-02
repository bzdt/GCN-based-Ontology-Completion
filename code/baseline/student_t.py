import numpy as np
from scipy.stats import t


# Train student t distribution for single concept
# output: parameters of student t distribution
# loc=param[1], scale=param[2], df=param[0]
def train_single_distribution(data):
    dim = np.array(data).shape[1]
    params = []      # dim rows, each for a dimension
    for i in range(dim):
        X = data[:,i]
        params.append(t.fit(X))  # scipy.stats
    return np.array(params)


# Train student t distribution for all concepts
def train_distribution(X, y):
    params = []
    for c in range(y.shape[1]):
        ind = np.where(y[:, c] == 1)  # find current concept
        if ind[0].size == 0:
            params.append([None,None,None])
            continue
        X_c = X[ind]  # training data for current concept
        params.append(train_single_distribution(X_c))  # parameters for current concept
    return params

# Compute probability for single concept
# params: parameters for current concept
# output: pdf
def compute_pdf_single(X, params):
    dim = np.array(X).shape[1]
    pdf = np.zeros(np.array(X).shape[0])
    for i in range(dim):
        p_i = t.pdf(X[:,i], loc=params[i,1], scale=params[i,2], df=params[i,0])
        pdf += p_i
    return pdf / dim


# Compute probability for all concepts
def compute_pdf(X, params, concept_num):
    pdf = np.zeros((X.shape[0], concept_num))
    for c in range(concept_num):
        if params[c][0] is None:
            continue
        pdf[:,c] = compute_pdf_single(X, params[c])
    return pdf

# Gaussian distribution
def train_gaussian(X):
    return np.array([np.mean(X, axis=0), np.var(X, axis=0)])

def compute_pdf_gaussian(X, params):
    mean = params[0]
    var = params[1]
    pdf = np.exp(- np.power((X - mean), 2) / (2 * var)) / (np.sqrt(2 * np.pi * var))
    return pdf.sum(axis=1)