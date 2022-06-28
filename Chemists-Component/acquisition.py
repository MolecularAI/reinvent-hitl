import numpy as np
from scipy.stats import norm
import time



def random_sampling(U, X_train, n, model, acquisition, X_updated, y_updated, L=None,t=None):
    ret = np.random.choice(U, n, replace=False)
    return ret

def uncertainty_sampling(U, X_train, n, model, acquisition, X_updated, y_updated, L, t=None):
    X_pool = X_train[U,:]
    preds_p, var = model.predict_f(X_pool)
    idx = np.argsort(np.squeeze(var))[-n:]
    return U[idx]

# exploitation
def greedy_algorithm(U, X_train, n, model, acquisition, X_updated, y_updated, L=None, t=None):
    X_pool = X_train[U,:]
    preds_p,_ = model.predict_f(X_pool)
    preds_p=np.squeeze(preds_p)
    idx = np.argsort(preds_p)[::-1]
    assert len(idx)>0
    selected = idx[:n]
    return U[selected]


def thompson_sampling(U, X_U, n, model, acquisition, X_updated, y_updated, L=None, t=None):
    X_pool =X_U[U,:]
    # a sample from posterior
    preds_p = model.predict_f_samples(X_pool) 
    preds_p = np.squeeze(preds_p)
    # greedily maximize with respect to the randomly sampled belief
    idx = np.argsort(preds_p)[::-1] 
    assert len(idx)>0
    selected = idx[:n]
    return U[selected]

def select_query(U, X_train, n, model, acquisition='random', X_updated=None, y_updated=None, L=None, t=None):
    '''
    Parameters
    ----------
    U : Indices of X_train that are unlabeled.
    X_train : Original training data (labeled and unlabeled). Current pool of unlabeled samples: X_train[U,:]
    n : number of queries to select.
    model : Current prediction model.
    acquisition : acquisition method, optional. The default is 'random'.

    Returns
    -------
    int idx: 
        Index of the query; idx \in U. Features of the query: X_train[idx,:]

    '''
    # select acquisition:
    if acquisition == 'uncertainty':
        acq = uncertainty_sampling
    elif acquisition == 'thompson':
        acq = thompson_sampling
    elif acquisition == 'random':
        acq = random_sampling
    elif acquisition == 'greedy':
        acq = greedy_algorithm
    else:
        print("Warning: Unknown acquisition function. Using random sampling.")
        acq = random_sampling
    return acq(U, X_train, n, model, acquisition, X_updated, y_updated, L, t)