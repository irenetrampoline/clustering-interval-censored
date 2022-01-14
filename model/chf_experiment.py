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


def make_test_prime(test_data_dict_raw, gap=1):
    test_data_dict = copy.deepcopy(test_data_dict_raw)
    eps_lst        = list()
    
    X = test_data_dict['obs_t_collect']
    Y = test_data_dict['Y_collect']
    M = test_data_dict['mask_collect']
    
    N_patients = X.shape[0]
    N_visits   = X.shape[1]
    
    for i in range(N_patients):
        eps_i = X[i,1,0] - X[i,0,0]
        
        first_visit = X[i,1,0]
        # move all visits down (essentially destroying the first visit)
        for j in range(N_visits-gap):
            
            X[i,j,0] = X[i,j+gap,0] - first_visit
            Y[i,j,:] = Y[i,j+gap,:]
            M[i,j,:] = M[i,j+gap,:]
        
        for g in range(1,gap+1):
            X[i,N_visits-g,0] = -1000
            Y[i,N_visits-g,:] = -1000
            M[i,N_visits-g,:] = 0
        
        eps_lst.append(eps_i)
    return test_data_dict, eps_lst

# TODO: load chf data
# TODO: train model / load trained CHF model (optional)
# TODO: split into train/test CHF
# TODO: made late-arrivat test CHF data

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--chf', action='store_true', help="Use CHF dataset")
    parser.add_argument('--gap', action='store', type=int, help="How many visits to destroy", default=1)
    args = parser.parse_args()
    
    if args.chf:
        data_type = 'chf'
    else:
        data_type = 'data1'
        
    if data_type =='chf':
        data       = chf()
        max_visits = 38
        shuffle    = True
        num_output_dims = data.shape[1] - 4

        data_loader, collect_dict, unique_pid = parse_data(data.values, max_visits=max_visits)
        train_data_loader, train_data_dict, test_data_loader, test_data_dict, test_pid, unique_pid = parse_data(data.values, max_visits=max_visits, test_per=0.2, shuffle=shuffle)

        model = Sublign(10, 20, 50, dim_biomarkers=num_output_dims, sigmoid=True, reg_type='l1', auto_delta=True, max_delta=5, learn_time=True)
#         model.fit(data_loader, data_loader, args.epochs, 0.01, verbose=args.verbose,fname='runs/chf.pt',eval_freq=25)
            
        fname='chf_good.pt'
        model.load_state_dict(torch.load(fname))
            
        # TODO: fix this
#         if fname is not None and epochs > eval_freq:
#             print('loaded state_dict. nelbo: %.4f (ep %d)' % (best_nelbo, best_ep))
#             model = Sublign()
#             model.load_state_dict(torch.load(fname))
#             model.eval()
    else:
        data_format_num = 1
        C, d_s, d_h, d_rnn, reg_type, lr = get_hyperparameters(data_format_num)
        data = load_data_format(data_format_num, 0, cache=True)

        train_data_loader, train_data_dict, _, _, test_data_loader, test_data_dict, valid_pid, test_pid, unique_pid = parse_data(data.values, max_visits=4, test_per=0.2, valid_per=0.2, shuffle=False)

        model  = Sublign(d_s, d_h, d_rnn, dim_biomarkers=3, sigmoid=True, reg_type='l1', auto_delta=True, max_delta=5, learn_time=True, b_vae=0.001)
#         model.fit(train_data_loader, test_data_loader, 800, lr, fname='runs/data%d_chf_experiment.pt' % (data_format_num), eval_freq=25)
        
        fname='runs/data%d_chf_experiment.pt' % (data_format_num)
        model.load_state_dict(torch.load(fname))
        results = model.score(train_data_dict, test_data_dict)
        print('ARI: %.3f' % results['ari'])

        # load data 1
        
        # train data 1 model / load data 1 model

    test_p_data_dict, eps_lst = make_test_prime(test_data_dict, gap=args.gap)

    
    test_deltas   = model.get_deltas(test_data_dict).detach().numpy()
    test_p_deltas = model.get_deltas(test_p_data_dict).detach().numpy()
    
    import pickle
    if data_type == 'chf':
        f = open('chf_experiment_results.pk','wb')
    else:
        f = open('data1_experiment_results.pk','wb')
    
    pickle.dump((test_deltas, test_p_deltas, eps_lst), f)
    f.close()
    
    est_esp = test_p_deltas - test_deltas
    
    pear  = pearsonr(eps_lst, est_esp)[0]
    swaps = swap_metrics(eps_lst, est_esp)
    per_pos = np.mean(np.array(test_p_deltas - test_deltas) > 0)
    
    print('Pear (up): %.3f, Swaps (down): %.3f, Per positive (up): %.3f' % (pear, swaps, per_pos))
    
    
#     import pdb; pdb.set_trace()
#     plt.plot(eps_lst, test_p_deltas - test_deltas, '.')
#     plt.savefig('')

if __name__=='__main__':
    main()