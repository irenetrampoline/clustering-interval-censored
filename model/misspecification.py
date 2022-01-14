import argparse
import numpy as np
import sys

sys.path.append('../data')
# from load import chf
from data_utils import parse_data
from synthetic_data import load_piecewise_synthetic_data


sys.path.append('../model')
from models import Sublign
from run_experiments import get_hyperparameters

sys.path.append('../baselines')
from kmeans import run_kmeans_baseline

parser = argparse.ArgumentParser() 
# parser.add_argument('--data_num', action='store', type=int, default=1, help="Data setting")
parser.add_argument('--increasing', action='store_true', help="Assume increasing?")
args = parser.parse_args()
    
N_epochs    = 800
N_trials    = 5
use_sigmoid = args.increasing

sublign_results = {
    'ari':[],
    'pear': [],
    'swaps': []
}
subnolign_results = {'ari': []}

kmeans_results    = {'ari': [], 
                     'pear': [],
                     'swaps': []}

for trial in range(N_trials):
    data_format_num = 1
    # C, d_s, d_h, d_rnn, reg_type, lr = get_hyperparameters(data_format_num)
    anneal, b_vae, C, d_s, d_h, d_rnn, reg_type, lr = get_hyperparameters(data_format_num)
    # C
    # data = load_data_format(data_format_num, 0, cache=True)

#     use_sigmoid = False

    data, subtype_points = load_piecewise_synthetic_data(subtypes=2, increasing=use_sigmoid, 
                            D=3, N=2000,M=4, noise=0.25, N_pts=5)

    train_data_loader, train_data_dict, _, _, test_data_loader, test_data_dict, valid_pid, test_pid, unique_pid = parse_data(data.values, max_visits=4, test_per=0.2, valid_per=0.2, shuffle=False)

    model  = Sublign(d_s, d_h, d_rnn, dim_biomarkers=3, sigmoid=use_sigmoid, reg_type='l1', 
                     auto_delta=False, max_delta=5, learn_time=True, beta=1.)
    model.fit(train_data_loader, test_data_loader, N_epochs, lr, fname='runs/data%d_spline.pt' % (data_format_num), eval_freq=25)

    # z = model.get_mu(train_data_dict['obs_t_collect'], train_data_dict['Y_collect'])
    # fname='runs/data%d_chf_experiment.pt' % (data_format_num)
    # model.load_state_dict(torch.load(fname))
    results = model.score(train_data_dict, test_data_dict)
    print('Sublign results: ARI: %.3f; Pear: %.3f; Swaps: %.3f' % (results['ari'],results['pear'],results['swaps']))
    sublign_results['ari'].append(results['ari'])
    sublign_results['pear'].append(results['pear'])
    sublign_results['swaps'].append(results['swaps'])
    
    model  = Sublign(d_s, d_h, d_rnn, dim_biomarkers=3, sigmoid=use_sigmoid, reg_type='l1', 
                     auto_delta=False, max_delta=0, learn_time=False, beta=1.)
    model.fit(train_data_loader, test_data_loader, N_epochs, lr, fname='runs/data%d_spline.pt' % (data_format_num), eval_freq=25)
    nolign_results = model.score(train_data_dict, test_data_dict)
    print('SubNoLign results: ARI: %.3f' % (nolign_results['ari']))
    subnolign_results['ari'].append(nolign_results['ari'])
    
    
    kmeans_mse, kmeans_ari, kmeans_swaps, kmeans_pear = run_kmeans_baseline(train_data_dict, test_data_dict, use_sigmoid=use_sigmoid)
    kmeans_results['ari'].append(kmeans_ari)
    kmeans_results['pear'].append(kmeans_pear)
    kmeans_results['swaps'].append(kmeans_swaps)
    
data_str = 'Increasing' if use_sigmoid else 'Any'
print('SubLign-%s & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f \\\\' % (
    data_str,
    np.mean(sublign_results['ari']), np.std(sublign_results['ari']),
    np.mean(sublign_results['pear']), np.std(sublign_results['pear']),
    np.mean(sublign_results['swaps']), np.std(sublign_results['swaps'])
))

print('SubNoLign-%s & %.2f $\\pm$ %.2f & -- &  --  \\\\' % (
    data_str,
    np.mean(sublign_results['ari']), np.std(sublign_results['ari']),
))

print('KMeans+Loss-%s & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f \\\\' % (
    data_str,
    np.mean(kmeans_results['ari']), np.std(kmeans_results['ari']),
    np.mean(kmeans_results['pear']), np.std(kmeans_results['pear']),
    np.mean(kmeans_results['swaps']), np.std(kmeans_results['swaps'])
))
