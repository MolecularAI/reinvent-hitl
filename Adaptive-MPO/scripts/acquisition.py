# Acquisitions for HITL interaction

import numpy as np

def local_idx_to_fulldata_idx(N, selected_feedback, idx):
    all_idx = np.arange(N)
    mask = np.ones(N, dtype=bool)
    mask[selected_feedback] = False
    pred_idx = all_idx[mask]
    return pred_idx[idx]

def uncertainty_sampling(N,n,fit,selected_feedback, rng, t=None):
    la = fit.extract(permuted=True)
    score_pred = np.mean(la['score_pred'],axis=0)
    utility = np.absolute(score_pred - 0.5)
    query_idx = np.argsort(utility)[:n] 
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)

def posterior_sampling(N,n,fit,selected_feedback, rng, t=None):
    print("Posterior sampling at iteration t={}".format(t))
    la = fit.extract(permuted=True)
    n_samples = la['score_pred'].shape[0] # number of posterior draws
    sample_idx = rng.choice(n_samples, n, replace=True) # draw n random beliefs from the posterior, with replacement
    query_idx = np.zeros(n) 
    for i in np.arange(0,len(sample_idx)):
        score_pred = la['score_pred'][sample_idx[i],:]
        cq = np.random.choice(np.flatnonzero(score_pred == score_pred.max())) # breaks ties at random
        k = len(score_pred)-1
        while(cq in query_idx[:i]):
            k = k-1 # take second best option
            if np.sum(score_pred == score_pred.max()) > 1:
                print("More than one maximizer")
            cq = np.argsort(score_pred)[k] # take the k:th highest value
        query_idx[i] = cq
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx.astype(int))

def exploitation(N,n,fit,selected_feedback, rng, t=None):
    la = fit.extract(permuted=True)
    score_pred = np.mean(la['score_pred'],axis=0)
    query_idx = np.argsort(score_pred)[::-1][:n] # get the n highest
    return local_idx_to_fulldata_idx(N, selected_feedback, query_idx)

def random_selection(N,n,fit,selected_feedback, rng, t=None):
    selected = rng.choice(N-len(selected_feedback),n, replace=False)
    return local_idx_to_fulldata_idx(N, selected_feedback, selected)


def select_query(N,n,fit,selected_feedback, acquisition='random', rng=None, t=None):
    '''
    Parameters
    ----------
    N : Size of the unlabeled data before acquisition
    n : number of queries to select.
    model : Current prediction model.
    acquisition : acquisition method, optional. The default is 'random'
    rng : random generator to be used

    Returns
    -------
    int idx: 
        Index of the query

    '''
    # select acquisition:
    if acquisition == 'uncertainty':
        acq = uncertainty_sampling
    elif acquisition == 'thompson':
        acq = posterior_sampling
    elif acquisition == 'greedy':
        acq = exploitation
    elif acquisition == 'random':
        acq = random_selection
    else:
        print("Warning: unknown acquisition criterion. Using random sampling.")
        acq = random_selection
    return acq(N, n, fit, selected_feedback, rng, t)
