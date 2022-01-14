# This works on `my_env` environment. On machines, make sure to run `condaup`

# PAGA dependencies
from anndata import AnnData
# from scanpy.plotting._anndata import _prepare_dataframe
import scanpy.api as sc

import pandas as pd
import numpy as np
import sys
sys.path.append('../evaluation')
from sklearn.metrics import adjusted_rand_score

from eval_utils import get_cluster_swap_metric, get_cluster_pear_metric

# SAUCIE dependencies
# sys.path.append('./baselines/SAUCIE/')
# from SAUCIE.model import SAUCIE
# from SAUCIE.loader import Loader

def run_paga_baseline(data_num=1, trials=1):
    """
    https://github.com/dynverse/ti_paga/blob/master/run.py

    inputs
    ===
    counts (array N x D): obs values
    start_id (int): early cell 
    
    parameters
    ===
    n_neighbors (int): neighbors for kNN, 1-100, default 15
    resolution (int): louvain parameter, 0.1-10, default 1
    n_dcs (int): 0-40, default 15
    connectivity_cutoff (float): 0-1, default 0.05
    """
    
    
    paga_trial_results    = np.zeros((trials, 3))
    
    # find right resolution
    
    data = load_data_format(data_num,trial_num=1, cache=True)
    _, train_data_dict, _, test_data_dict, _, _ = parse_data(data.values, max_visits=4, test_per=0.2)

    # find right resolution on train_data_dict
    # binary search from 1. to 0. to figure out number_clusters
    cur_resolution = 1.
    old_resolution = cur_resolution
    n_clusters     = 0.
    
#     while n_clusters != 2:
#         pear, swaps, ari, n_clusters = run_paga_baseline_wrapper(train_data_dict, resolution=cur_resolution, searching_resolution=True)
#         print('Cur resolution: %f, N_clusters:%d' % (cur_resolution, n_clusters))
        
#         if n_clusters > 2:
#             cur_resolution = cur_resolution / 2.
#         else:
#             cur_resolution = (cur_resolution + 1) / 2.
        
    cur_resolution = 0.0001
    print('Optimal resolution: %.4f' % cur_resolution)

    max_visits = 4
    if data_num > 10:
        max_visits = 17
        
    for trial_num in range(trials):
        data = load_data_format(data_num,trial_num=trial_num, cache=True)
        _, train_data_dict, _, test_data_dict, _, _ = parse_data(data.values, max_visits=max_visits, test_per=0.2)
        
        pear, swaps, ari = run_paga_baseline_wrapper(train_data_dict, resolution=cur_resolution)
        paga_trial_results[trial_num] = [ari, swaps, pear]
        
    if trials == 1:
        print('%s, --, %.3f, %.3f, %.3f' % ('PAGA', ari, swaps, pear))
    else:
        line_str = list()
        for i,j in zip(paga_trial_results.mean(axis=0), paga_trial_results.std(axis=0)): 
            line_str.append('%.3f $\\pm$ %.3f' % (i,j))
        print(' & '.join(['%d: PAGA' % data_num] + line_str) + '\\\\')
        
    
def run_paga_baseline_wrapper(data_dict, resolution=0.028, searching_resolution=False):
    # cluster by raw values, then run paga, then report within-cluster
    X = data_dict['Y_collect']
    N_patients, N_visits, N_dims = X.shape
    X = np.reshape(X,(N_patients * N_visits, N_dims))
    
    t = data_dict['t_collect']
    t = np.reshape(t,(N_patients * N_visits, 1))

    start_id = str(np.argmin(t))
    if searching_resolution:
        pseudotimes, clusters, n_clusters = run_paga(X,start_id, resolution, show_num_clusters=searching_resolution)
    else:
        pseudotimes, clusters = run_paga(X,start_id, resolution)
    
    real_t = data_dict['t_collect'][:,:,:].flatten().squeeze()
    
    labels = clusters

    gt_subtypes = [int(i) for i in data_dict['s_collect'].flatten()]
    gt_subtypes = np.concatenate([[i] * N_visits for i in gt_subtypes])
    
    pear   = get_cluster_pear_metric(labels, pseudotimes, real_t)
    swaps  = get_cluster_swap_metric(labels, pseudotimes, real_t)
    ari   = adjusted_rand_score(labels, gt_subtypes)
        
#     print('ARI: %.3f' % ari)
#     print('Pearson: %.3f' % pear)
#     print('Swaps: %.3f' % swaps)

    if searching_resolution:
        return pear, swaps, ari, n_clusters
    else:
        return pear, swaps, ari
    
def run_paga(X, start_id, resolution, show_num_clusters=False):
#     print('Running PAGA with resolution %f' % resolution)
    if X.shape[1] == 1:
        # hack to handle case where dim=1
        X = np.hstack([X,np.ones(X.shape)])
    n_dcs = 10 
    n_neighbors = 15
    resolution = 1.
    connectivity_cutoff = 0.05
    
    adata = AnnData(X)

    sc.pp.normalize_total(adata)
    sc.pp.scale(adata)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.diffmap(adata, n_comps=n_dcs)

    sc.tl.louvain(adata, resolution=resolution)
    sc.tl.paga(adata)
    
    sc.pl.paga(adata, threshold=0.01, layout='fr', show=False)
    adata.uns['iroot'] = np.where(adata.obs.index == start_id)[0][0]
    sc.tl.dpt(adata, n_dcs = min(adata.obsm['X_diffmap'].shape[1], 10))
    sc.tl.umap(adata, init_pos='paga')

    grouping = pd.DataFrame({"cell_id": adata.obs.index, "group_id": adata.obs.louvain})
    
    # milestone network
    milestone_network = pd.DataFrame(
      np.triu(adata.uns["paga"]["connectivities"].todense(), k = 0),
      index=adata.obs.louvain.cat.categories,
      columns=adata.obs.louvain.cat.categories
    ).stack().reset_index()
    milestone_network.columns = ["from", "to", "length"]
    milestone_network = milestone_network.query("length >= " + str(connectivity_cutoff)).reset_index(drop=True)
    milestone_network["directed"] = False
    
    # dimred
    dimred = pd.DataFrame([x for x in adata.obsm['X_umap'].T]).T
    dimred.columns = ["comp_" + str(i) for i in range(dimred.shape[1])]
    dimred["cell_id"] = adata.obs.index

    # branch progressions: the scaled dpt_pseudotime within every cluster
    branch_progressions = adata.obs
    branch_progressions["dpt_pseudotime"] = branch_progressions["dpt_pseudotime"].replace([np.inf, -np.inf], 1) # replace unreachable pseudotime with maximal pseudotime
    branch_progressions["percentage"] = branch_progressions.groupby("louvain")["dpt_pseudotime"].apply(lambda x: (x-x.min())/(x.max() - x.min())).fillna(0.5)
    branch_progressions["cell_id"] = adata.obs.index
    branch_progressions["branch_id"] = branch_progressions["louvain"].astype(np.str)
    branch_progressions = branch_progressions[["cell_id", "branch_id", "percentage"]]

    # branches:
    # - length = difference between max and min dpt_pseudotime within every cluster
    # - directed = not yet correctly inferred
    branches = adata.obs.groupby("louvain").apply(lambda x: x["dpt_pseudotime"].max() - x["dpt_pseudotime"].min()).reset_index()
    branches.columns = ["branch_id", "length"]
    branches["branch_id"] = branches["branch_id"].astype(np.str)
    branches["directed"] = True

    # branch network: determine order of from and to based on difference in average pseudotime
    branch_network = milestone_network[["from", "to"]]
    average_pseudotime = adata.obs.groupby("louvain")["dpt_pseudotime"].mean()
    for i, (branch_from, branch_to) in enumerate(zip(branch_network["from"], branch_network["to"])):
        if average_pseudotime[branch_from] > average_pseudotime[branch_to]:
            branch_network.at[i, "to"] = branch_from
            branch_network.at[i, "from"] = branch_to

    pseudotime = adata.obs['dpt_pseudotime'].values
    clusters   = adata.obs.louvain.values.get_values()
    n_clusters = len(np.unique(clusters))
    if n_clusters != 2:
        print('Number clusters: %d' % n_clusters)
        
    if show_num_clusters:
        return pseudotime, clusters, n_clusters
    else:
        return pseudotime, clusters

def run_saucie_baseline(trials=1):
    """
    https://github.com/dynverse/ti_paga/blob/master/run.py

    inputs
    ===
    counts (array N x D): obs values
    start_id (int): early cell 
    
    parameters
    ===
    n_neighbors (int): neighbors for kNN, 1-100, default 15
    resolution (int): louvain parameter, 0.1-10, default 1
    n_dcs (int): 0-40, default 15
    connectivity_cutoff (float): 0-1, default 0.05
    """
    
    
    saucie_trial_results    = np.zeros((trials, 1))
    
    for trial_num in range(trials):
        data = load_data_format(1,trial_num=trial_num, cache=True)
        _, train_data_dict, _, test_data_dict, _, _ = parse_data(data.values, max_visits=4, test_per=0.2)
        ari = run_saucie(train_data_dict)
        saucie_trial_results[trial_num] = [ari]
        
    if trials == 1:
        print('%s, --, %.3f, --, --' % ('PAGA', ari))
    else:
        line_str = list()
        for i,j in zip(saucie_trial_results.mean(axis=0), saucie_trial_results.std(axis=0)): 
            line_str.append('%.3f $\\pm$ %.3f' % (i,j))
        print(' & '.join(['SAUCIE'] + line_str) + '\\\\')
        
def run_saucie(data_dict):
    x      = data_dict['Y_collect']
    real_t = data_dict['t_collect'][:,0,:].flatten()
    
    N_patients, N_visits, N_dims = x.shape
    x = np.reshape(x,(N_patients * N_visits, N_dims))
    
    load = Loader(x, shuffle=False)
    
    saucie = SAUCIE(x.shape[1], lambda_c=.2, lambda_d=.4)
    saucie.train(load, 200)
    embedding = saucie.get_embedding(load)
    num_clusters, clusters = saucie.get_clusters(load)
    
    true_clusters = np.tile(data_dict['s_collect'], (1,N_visits))
    true_clusters = np.reshape(true_clusters, (N_patients * N_visits,1)).squeeze()
    
    
    start_id = str(np.argmin(real_t))
    
#     labels     = np.ones(N_patients)
#     pear       = get_cluster_pear_metric(labels, real_t, pseudotimes)
#     swaps      = get_cluster_swap_metric(labels, real_t, pseudotimes)

    ari   = adjusted_rand_score(true_clusters, clusters)
    
    print('ARI: %.3f' % ari)
#     print('Pearson: %.3f' % pear)
#     print('Swaps: %.3f' % swaps)
    
    return ari


if __name__=='__main__':
    import sys
    sys.path.append('../data/')
    from load import load_data_format
    from data_utils import parse_data
    
    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument('--paga', action='store_true', default=True, help="Use PAGA")
    parser.add_argument('--saucie', action='store_true', help="Use SAUCIE")
    parser.add_argument('--trials', action='store', type=int, default=1, help="Number of data versions to run (data_num=1)")
    parser.add_argument('--data_num', action='store', type=int, default=1, help="Data num to run")
    args = parser.parse_args()
    
    
    if args.paga:
        run_paga_baseline(trials=args.trials, data_num=args.data_num)
    else:
        run_saucie_baseline(trials=args.trials)
