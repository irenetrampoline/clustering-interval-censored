import numpy as np
import torch
import sys, os
sys.path.append('../data/')
sys.path.append('../model/')
from load import sigmoid, quadratic, load_data_format, parkinsons
from load import chf as load_chf
from data_utils import parse_data
from models_discrete import SublignDiscrete
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def run_model(MODEL, hp, lr, train_loader, train_data_dict, valid_loader, valid_data_dict, device, epochs = 1000, eval_freq = 25, ppmi=False, chf=False, fname=None, search='nelbo', anneal=False):
#     print ('Running for ',epochs, ' epochs w/ evfreq',eval_freq)
    model = MODEL(**hp)
    model.to(device)
    
    model.fit(train_loader, valid_loader, epochs, lr, eval_freq=eval_freq, fname=fname)
    results = model.score(train_data_dict,valid_data_dict, K=2)
    
    sublign_ari = results['ari']
    sublign_swaps = results['swaps']
    sublign_pear = results['pear']
    
    train_Y = train_data_dict['Y_collect']
    train_X = train_data_dict['obs_t_collect']
    
    test_Y = valid_data_dict['Y_collect']
    test_S = valid_data_dict['s_collect']
    test_X = valid_data_dict['obs_t_collect']
    test_M = valid_data_dict['mask_collect']
    
    test_X_torch = torch.tensor(test_X).to(device)
    test_Y_torch = torch.tensor(test_Y).to(device)
    test_M_torch = torch.tensor(test_M).to(device)
        
    
    align_metrics = np.mean([1-sublign_swaps, sublign_pear])
    dummy = torch.ones_like(test_Y_torch)
    
    nelbo = model.forward(test_Y_torch, dummy, test_X_torch, test_M_torch, dummy)
    
    nelbo_metric  = nelbo
    cheat_metric  = - np.mean([sublign_ari, align_metrics])
    
    # We are MINIMIZING over metrics
    if search == 'cheat':
        final_metric = cheat_metric
    elif search == 'nelbo':
        final_metric = nelbo_metric
    
    all_metrics = {
        'nelbo': nelbo_metric,
        'cheat': cheat_metric,
        'ari': sublign_ari,
        'pear': sublign_pear,
        'swaps': sublign_swaps
    }
    return final_metric, all_metrics

def get_unsup_results(train_loader, train_data_dict, valid_loader, test_data_dict, device, model, epochs, sigmoid, ppmi=False, chf=False, fname=None, search='nelbo'):
    """
    We are looking for the LOWEST metric over all the parameter searches. Make sure metrics are tuned accordingly!
    
    """
    hp = {}
    best_perf, best_mse, best_ari, best_swaps, best_pear = np.inf, np.inf, np.inf, np.inf, np.inf
    best_config = None
    all_results = {}
    
    MODEL = None
    if model == 'sublign':
        MODEL = SublignDiscrete
    else:
        NotImplemented()
    print (MODEL)
                 
        
#     anneal, beta, C, ds, dh, drnn, reg_type, lr = True, 0.001, 0.0, 5, 200, 200, 'l2', 0.001
        
    if model =='sublign':
        for K in [2,4,6]:
            for lr in [1e-3, 1e-2, 1e-1]:  
                hp['K']           = K
                perf, all_metrics = run_model(MODEL, hp, lr, train_loader, train_data_dict, valid_loader, 
                                                  test_data_dict, device, epochs = epochs, ppmi=ppmi, 
                                                  eval_freq=1, fname=fname, search=search)
                print((K, lr, all_metrics['ari']))
                all_results[(K, lr)] = all_metrics
                if perf<best_perf:
                    best_perf = perf; best_ari = all_metrics['ari']; best_config = (K, lr)
    return best_perf, best_ari, best_config, all_results

def run_cv(model = 'sublign', dataset = 'sigmoid', epochs=10, data_num=1, ppmi=False, chf=False, search='nll'):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    device  = torch.device('cpu')
    
    print ('Running on: ', device)
    
    # load data
    if ppmi:
        use_sigmoid = True
        data = parkinsons()
        max_visits = 17
        train_loader, train_data_dict, valid_loader, valid_dict, p_ids, full_p_ids = parse_data(data.values, max_visits=max_visits, test_per=0.2)
        fname = 'runs/ppmi_hptune.pt'
    if chf:
        use_sigmoid = True
        data = load_chf()
        max_visits = 38
        train_loader, train_data_dict, valid_loader, valid_dict, p_ids, full_p_ids = parse_data(data.values, max_visits=max_visits, test_per=0.2)
        fname = 'runs/chf_hptune.pt'
    else:
        use_sigmoid = data_num < 3
        data        = load_data_format(data_num, 0)
        train_loader, train_data_dict, valid_loader, valid_dict, _, _, valid_pid, test_pid, unique_pid = parse_data(data.values, max_visits=4, test_per=0.2, valid_per=0.2, shuffle=False)
        fname = 'runs/data%d_trial%d_hptune_0611.pt' % (data_num, 0)
    
    # run hyperparamter search
    best_perf, best_ari, best_config, all_results = get_unsup_results(train_loader, train_data_dict, valid_loader, valid_dict, device, model, epochs=epochs, sigmoid=use_sigmoid, ppmi=ppmi, chf=chf,fname=fname, search=search)

    print(best_config, 'Best ARI: %.3f' % best_ari)
#     print ('Done fold ',fold)
    
    print ('Saving...')
#     suffix = '.'.join([str(k) for k in fold_span])

    if ppmi:
        fname = 'discrete_runs/'+model+'_ppmi.pkl'
    elif chf:
        fname = 'discrete_runs/'+model+'_chf_%d.pkl' % epochs
    else:
        fname = 'discrete_runs/'+model+'_'+str(data_num) + '_0309'+ '_'+ str(epochs) +'.pkl'
    
    with open(fname,'wb') as f:
        print('dumped pickle')
        pickle.dump(all_results, f)
        
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument('--epochs', action='store', type=int, default=800, help="Number of epochs")
    parser.add_argument('--data_num', action='store', type=int, default=1, help="Data scenario number")
    parser.add_argument('--ppmi', action='store_true', help="Hyperparam search on PPMI data")
    parser.add_argument('--chf', action='store_true', help="Hyperparam search on CHF data")
    
    parser.add_argument('--kmeans', action='store_true', help="Hyperparam search based on kmeans ARI")
    parser.add_argument('--nelbo', action='store_true', help="Hyperparam search based on NELBO")
    parser.add_argument('--nll', action='store_true', help="Hyperparam search based on NLL")
    parser.add_argument('--cheat', action='store_true', help="Hyperparam search based on cheating")
    
    args = parser.parse_args()
        
    if args.cheat:
        search = 'cheat'
    elif args.nelbo:
        search = 'nelbo'
    else:
        search = 'nelbo'
    
    run_cv(model = 'sublign', dataset='sigmoid', epochs=args.epochs, data_num=args.data_num, ppmi=args.ppmi, chf=args.chf, search=search)