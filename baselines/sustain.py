import sys
sys.path.append('../../pySuStaIn/')
sys.path.append('../evaluation/')

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import sklearn.model_selection
import pandas as pd
import pylab

# import the simulation functions from pySuStaIn needed to generate simulated data
from simfuncs import generate_random_sustain_model, generate_data_sustain

# import the functions for z-score SuStaIn
from ZscoreSustain  import ZscoreSustain

from eval_utils import get_cluster_pear_metric, get_cluster_swap_metric

from sklearn.metrics import adjusted_rand_score

def run_sustain_baseline(epochs, optim_epochs, trials, ppmi=False, data_num=1, how_impute='mice'):
    sustain_trial_results    = np.zeros((trials, 3))
    
    for trial_num in range(trials):
        ari, swaps, pear = run_sustain_notebook(epochs, optim_epochs, trial_num, ppmi, data_num=data_num, how_impute=how_impute)
        sustain_trial_results[trial_num] = [ari, swaps, pear]
        
    if trials == 1:
        print('%s, %s, %.3f, %.3f, %.3f' % ('SuStaIn', '--', ari, swaps, pear))
    else:
        line_str = list()
        for i,j in zip(sustain_trial_results.mean(axis=0), sustain_trial_results.std(axis=0)): 
            line_str.append('%.3f $\\pm$ %.3f' % (i,j))
        print(' & '.join(['SuStaIn'] + line_str) + '\\\\')
        
    import os.path
    
    trials_fname = '../runs/sustain%d_experiments.txt' % trials

    f = open(trials_fname, 'wb')
    import pickle
    pickle.dump(sustain_trial_results, f)
    f.close()
    
    print(sustain_trial_results)

def run_sustain_notebook(epochs=100, optim_epochs=1000, trial_num=1,ppmi=False, data_num=1, how_impute='mice'):
    epochs       = int(epochs)
    optim_epochs = int(optim_epochs)
    
    N                       = 5         # number of biomarkers
    M                       = 500       # number of observations ( e.g. subjects )
    M_control               = 100       # number of these that are control subjects
    N_S_gt                  = 2         # number of ground truth subtypes

    N_startpoints = 10
    N_S_max = 2
    n_iterations_MCMC_optimisation = optim_epochs # replace to 1e4
    N_iterations_MCMC = epochs
    output_folder = 'data1'
    dataset_name = 'data1'
    
    # LOAD SUBLIGN SYNTHETIC DATA
        
    import sys
    sys.path.append('../data/')
    from load import load_data_format
    from data_utils import parse_data

    if ppmi:
        from load import parkinsons
        data = parkinsons()
        max_visits = 17
    else:
        data = load_data_format(data_num,trial_num=trial_num, cache=True)
        max_visits = 4 if data_num < 10 else 17
        
    _, train_data_dict, _, test_data_dict, _, _ = parse_data(data.values, max_visits=max_visits, test_per=0.2)
    
    if data_num == 11 or data_num == 12 or data_num == 14:
        X = train_data_dict['obs_t_collect']
        Y = train_data_dict['Y_collect']
        M = train_data_dict['mask_collect']

        X[X == -1000] = np.nan
        Y[Y == -1000] = np.nan

        sys.path.append('../model')
        from utils import interpolate

        if how_impute == 'mrnn':
            Y[np.isnan(Y)] = 0.
            X[np.isnan(X)] = 0.
            
        Y_impute = interpolate(Y, m=M, t=X, how=how_impute)
        
        train_data_dict['Y_collect'] = Y_impute
        
        X = test_data_dict['obs_t_collect']
        Y = test_data_dict['Y_collect']
        M = test_data_dict['mask_collect']

        X[X == -1000] = np.nan
        Y[Y == -1000] = np.nan

        sys.path.append('../model')
        from utils import interpolate

        if how_impute == 'mrnn':
            Y[np.isnan(Y)] = 0.
            X[np.isnan(X)] = 0.
            
        Y_impute = interpolate(Y, m=M, t=X, how=args.how_impute)
        
        test_data_dict['Y_collect'] = Y_impute
            
    N_patients, N_visits, N_dims = train_data_dict['Y_collect'].shape
    
    gt_stages   = train_data_dict['t_collect'].reshape((N_patients * N_visits, 1))
    gt_subtypes = [int(i) for i in train_data_dict['s_collect'].flatten()]
    gt_subtypes = np.concatenate([[i] * max_visits for i in gt_subtypes])
    data      = train_data_dict['Y_collect'].reshape((N_patients * N_visits, N_dims))

    mean_data = np.mean(data,axis=0) - 0.1
    std_data = np.std(data,axis=0) / 2

    data = (data-mean_data)/std_data

    print('Mean of whole dataset is ',np.mean(data,axis=0))
    # Check that the standard deviation of the whole dataset is greater than 1
    print('Standard deviation of whole dataset is ',np.std(data,axis=0))

    print(data.max())
    print(data.min())

    # size (N_biomarkers, N_subtypes)
    Z_vals = np.array([[1,2,3]]*N_dims)
    print(Z_vals)

    # size (N_biomarkers, )
    Z_max = np.array([5]*N_dims)
    print(Z_max)

    # Titles of dimensions
    SuStaInLabels = ['dim_%d' % i for i in range(1,N_dims+1)]
    print(SuStaInLabels)

    
    sustain_input = ZscoreSustain(data,
                                  Z_vals,
                                  Z_max,
                                  SuStaInLabels,
                                  N_startpoints,
                                  N_S_max, 
                                  N_iterations_MCMC, 
                                  n_iterations_MCMC_optimisation,
                                  output_folder, 
                                  dataset_name, 
                                  False)
    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        
    sustain_input.run_sustain_algorithm()
    
    pred_subtypes = sustain_input.ml_subtype
    pred_stages   = sustain_input.ml_stage
    
    pred_subtypes = pred_subtypes.squeeze()
    pred_stages   = pred_stages.astype(float).squeeze()
    gt_stages     = gt_stages.squeeze()
    
    def evaluate_sustain(pred_subtypes, gt_subtypes, gt_stages, pred_stages):
        pear  = get_cluster_pear_metric(pred_subtypes, gt_stages, pred_stages)
        swaps = get_cluster_swap_metric(pred_subtypes, gt_stages, pred_stages)
        ari   = adjusted_rand_score(pred_subtypes, gt_subtypes)
        return pear, swaps, ari
    
    
    pear, swaps, ari = evaluate_sustain(pred_subtypes, gt_subtypes, gt_stages, pred_stages)
    
    # MSE
    # gt_stage_value[b,:,:]

    print('Train ARI: %.3f' % ari)
    print('Train Swaps: %.3f' % swaps)
    print('Train Pear: %.3f' % pear)    
    
    samples_sequence = sustain_input.samples_sequence
    samples_f        = sustain_input.samples_f
    
    N_samples = samples_sequence.shape[2]
    
    # Make test_data
#     train_data_loader, train_data_dict, test_data_loader, test_data_dict, p_ids, full_p_ids = parse_data(data.values, max_visits=4, test_per=0.2)

    test_N_patients, test_N_visits, test_N_dims = test_data_dict['Y_collect'].shape
    
    test_gt_stages   = test_data_dict['t_collect'].reshape((test_N_patients * test_N_visits, 1))
    test_gt_subtypes = [int(i) for i in test_data_dict['s_collect'].flatten()]
    test_gt_subtypes = np.concatenate([[i] * max_visits for i in test_gt_subtypes])
    test_data      = test_data_dict['Y_collect'].reshape((test_N_patients * test_N_visits, test_N_dims))

    test_mean_data = np.mean(test_data,axis=0) - 0.1
    test_std_data = np.std(test_data,axis=0) / 2

    test_data = (test_data-test_mean_data)/test_std_data
    
    test_subtypes, _, test_stages, _, _, _, _ = sustain_input.subtype_and_stage_individuals_newData(test_data, samples_sequence, samples_f, N_samples)
    
    test_subtypes  = test_subtypes.squeeze()
    test_stages    = test_stages.astype(float).squeeze()
    test_gt_stages = test_gt_stages.squeeze()
    
    test_pear, test_swaps, test_ari = evaluate_sustain(test_subtypes, test_gt_subtypes, test_gt_stages, test_stages)
    
    print('Test ARI: %.3f' % test_ari)
    print('Test Swaps: %.3f' % test_swaps)
    print('Test Pear: %.3f' % test_pear) 
    return test_pear, test_swaps, test_ari

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument('--debug', action='store_true', help="DEBUG MODE")
    parser.add_argument('--trials', action='store', type=int, default=1, help="How many data sets to try (only data_num=1)")
    parser.add_argument('--ppmi', action='store_true', help="Use PPMI data")
    parser.add_argument('--data_num', action='store', type=int, default=1)
    parser.add_argument('--how_impute', action='store', default='mice')
#     parser.add_argument('--data_num', action='store', type=int, help="Data scenario number", default=1)
#     parser.add_argument('--trials', action='store', type=int, default=1, help="Number of trials")
#     parser.add_argument('--data_num', action='store', type=int, default=1, help="Data Format Number")
    args = parser.parse_args()
    
    if args.debug:
        run_sustain_baseline(epochs=100, optim_epochs=10, trials=args.trials, ppmi=args.ppmi, data_num=args.data_num)
#         run_sustain_notebook(epochs=100, optim_epochs=10, trial_num=1)
    else:
        print('WARNING: running in full')
        run_sustain_baseline(epochs=1e6, optim_epochs=1e4, trials=args.trials, ppmi=args.ppmi, data_num=args.data_num)
#         run_sustain_notebook(epochs=1e6, optim_epochs=1e4, trial_num=1)
    
    