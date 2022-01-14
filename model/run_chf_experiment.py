"""
semi-synthetic experiment

Alg:
Split data into train_data, test_data
train on train_data
alter test_data by dropping first (or first few) visits so that even if we don’t know the original delta, we know delta’ = delta + epsilon where epsilon is the time between the first and second visit. since we drop first visit, we expect that algorithm to learn a larger delta at least.
double test_data into a) original test_data with unknown delta, observed X and Y and b) altered test_data with unknown delta delta’, known epsilon, altered X’, Y’
plot (delta’ - delta) vs epsilon. I do not expect the match to be exactly y=x, but even some rough y=x trend could be interesting
"""
import argparse
import numpy as np
import pickle
import sys
import torch
import copy

from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from run_experiments import get_hyperparameters
from models import Sublign

sys.path.append('../data')

from data_utils import parse_data
from load import load_data_format, chf

sys.path.append('../evaluation')
from eval_utils import swap_metrics


def clean_plot():
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)    
    
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()   
    plt.grid()

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (10,6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


def make_test_prime(test_data_dict_raw, thresh=0.5):
    test_data_dict_new = copy.deepcopy(test_data_dict_raw)
    eps_lst        = list()

    X = test_data_dict_new['obs_t_collect']
    Y = test_data_dict_new['Y_collect']
    M = test_data_dict_new['mask_collect']

    N_patients = X.shape[0]
    N_visits   = X.shape[1]

    remove_idx = list()

    X[X == -1000] = np.nan

    for i in range(N_patients):
        N_visits_under_thresh = (X[i] < 0.5).sum()
        gap = N_visits_under_thresh

        first_valid_visit     = X[i,N_visits_under_thresh,0]

        eps_i = X[i,N_visits_under_thresh,0]

        for j in range(N_visits-N_visits_under_thresh):
            X[i,j,0] = X[i,j+gap,0] - first_valid_visit
            Y[i,j,:] = Y[i,j+gap,:]
            M[i,j,:] = M[i,j+gap,:]

        for g in range(1,N_visits_under_thresh+1):
            X[i,N_visits-g,0] = np.nan
            Y[i,N_visits-g,:] = np.nan
            M[i,N_visits-g,:] = 0.

        if np.isnan(X[i]).all():
            remove_idx.append(i)
        else:
            eps_lst.append(eps_i)

    keep_idx = [i for i in range(N_patients) if i not in remove_idx]
    X = X[keep_idx]
    Y = Y[keep_idx]
    M = M[keep_idx]

    print('Removed %d entries' % len(remove_idx))
    X[np.isnan(X)] = -1000
    return test_data_dict_new, eps_lst, keep_idx

# TODO: load chf data
# TODO: train model / load trained CHF model (optional)
# TODO: split into train/test CHF
# TODO: made late-arrivat test CHF data

def get_sublign(version=1, num_output_dims=12):
    if version == 0:
        ds, dh, drnn, reg_type, C, b_vae, lr, epochs, learn_time = 10, 20, 50,'l1', 0.0, 0.001, 0.01, 1000, False
        fname = 'runs/chf.pt'
    elif version == 1:
        ds, dh, drnn, reg_type, C, b_vae, lr, epochs, learn_time = 10,200,200,'l1', 0.0, 0.001, 0.01, 1000, True
        fname = 'runs/chf_v1_1000.pt'
    elif version == 2:
        ds, dh, drnn, reg_type, C, b_vae, lr, epochs, learn_time = 10,200,200,'l1', 0.1, 0.001, 0.01, 1000, True
        fname = 'runs/chf_v2_1000.pt'
    elif version == 3:
        ds, dh, drnn, reg_type, C, b_vae, lr, epochs, learn_time = 10,200,200,'l1', 0.0, 0.001, 0.1, 1000, True
        fname = 'runs/chf_v3_1000.pt'
        
    model = Sublign(ds, dh, drnn, dim_biomarkers=num_output_dims, sigmoid=True, reg_type=reg_type, auto_delta=False, C=C, b_vae=b_vae, 
                        max_delta=5, learn_time=True, device=torch.device('cpu'))
    
    model.load_state_dict(torch.load(fname,map_location=torch.device('cpu')))
    model = model.to(torch.device('cpu'))
    model.device = torch.device('cpu')
    return model
    
def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--chf', action='store_true', help="Use CHF dataset")
    parser.add_argument('--fresh', action='store_true', help="Run a fresh model (only for synthetic)")
    parser.add_argument('--thresh', action='store', type=float, help="What time under which visits are destroyed", default=0.5)
    parser.add_argument('--version', action='store', type=int, help="Which sublign", default=1)
    args = parser.parse_args()
    
    data       = chf()
    max_visits = 38
    shuffle    = True
    num_output_dims = data.shape[1] - 4
    
    data_loader, collect_dict, unique_pid = parse_data(data.values, max_visits=max_visits)
    model = get_sublign(version=2)
        
#     else:
#         data_format_num = 1
#         b_vae, C, d_s, d_h, d_rnn, reg_type, lr = get_hyperparameters(data_format_num)
#         data = load_data_format(data_format_num, 0, cache=True)

#         train_data_loader, train_data_dict, _, _, test_data_loader, test_data_dict, valid_pid, test_pid, unique_pid = parse_data(data.values, max_visits=4, test_per=0.2, valid_per=0.2, shuffle=False)

#         model  = Sublign(d_s, d_h, d_rnn, dim_biomarkers=3, sigmoid=True, reg_type='l1', auto_delta=True, max_delta=5, learn_time=True, b_vae=b_vae)
#         if args.fresh:
#             model.fit(train_data_loader, test_data_loader, 800, lr, fname='runs/data%d_chf_experiment.pt' % (data_format_num), eval_freq=25)
#         else:
#             fname='runs/data%d_chf_experiment.pt' % (data_format_num)
#             model.load_state_dict(torch.load(fname))
#         results = model.score(train_data_dict, test_data_dict)
#         print('ARI: %.3f' % results['ari'])

    trial_results = np.zeros((5, 3))
    for trial in range(5):
        train_data_loader, train_data_dict, test_data_loader, test_data_dict, test_pid, unique_pid = parse_data(data.values, max_visits=max_visits, test_per=0.2, shuffle=shuffle)

        test_p_data_dict, eps_lst, keep_idx = make_test_prime(test_data_dict, thresh=args.thresh)
    
#     import pdb; pdb.set_trace()
        test_deltas   = model.get_deltas(test_data_dict).detach().numpy()
        test_p_deltas = model.get_deltas(test_p_data_dict).detach().numpy()
    
        est_eps = np.array(test_p_deltas - test_deltas)
        est_eps = est_eps[keep_idx]

    #     import pdb; pdb.set_trace()
        pear  = pearsonr(eps_lst, est_eps)[0]
        swaps = swap_metrics(eps_lst, est_eps)
        per_pos = np.mean(np.array(test_p_deltas - test_deltas) > 0)

        trial_results[trial] = (pear, swaps, per_pos)
#         print('Pear (up): %.3f, Swaps (down): %.3f, Per positive (up): %.3f' % (pear, swaps, per_pos))
    line_str = list()
    for i,j in zip(trial_results.mean(axis=0), trial_results.std(axis=0)): 
        line_str.append('%.3f $\\pm$ %.3f' % (i,j))
    print(' & '.join(['CHF experiment'] + line_str) + '\\\\')
    
#     import pickle
#     if data_type == 'chf':
#         f = open('chf_experiment_results.pk','wb')
#     else:
#         f = open('data1_experiment_results.pk','wb')
    
#     results = {
#         'test_deltas': test_deltas,
#         'test_p_deltas': test_p_deltas,
#         'eps_lst': eps_lst,
#         'test_data_dict': test_data_dict
#     }
#     pickle.dump(results, f)
#     f.close()
    
    
    
    
#     import pdb; pdb.set_trace()
#     plt.plot(eps_lst, test_p_deltas - test_deltas, '.')
#     plt.savefig('')

if __name__=='__main__':
    main()