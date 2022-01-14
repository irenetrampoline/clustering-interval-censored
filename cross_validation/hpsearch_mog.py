import numpy as np
import torch
import sys, os
sys.path.append('../data/')
sys.path.append('../model/')
from load import sigmoid, quadratic, load_data_format, parkinsons
from load import chf as load_chf
from data_utils import parse_data
from models_mog import Sublign_MoG
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def run_model(MODEL, hp, lr, train_loader, train_data_dict, valid_loader, valid_data_dict, device, epochs = 1000, eval_freq = 25, ppmi=False, chf=False, fname=None, search='nll', anneal=False):
#     print ('Running for ',epochs, ' epochs w/ evfreq',eval_freq)
    model = MODEL(**hp)
    model.to(device)
    model.fit(train_loader, valid_loader, epochs, lr,anneal=anneal)
    
    Y = valid_data_dict['Y_collect']
    X = valid_data_dict['obs_t_collect']
    M = valid_data_dict['mask_collect']
    T = valid_data_dict['t_collect']
    S = valid_data_dict['s_collect']
    
    (norm_nelbo, norm_nll, norm_kl), norm_reg = model.forward(Y, S, X, M, T, anneal = 1.)
    all_metrics = {
        'nelbo': norm_nelbo,
        'nll': norm_nll,
        'kl': norm_kl
    }
    final_metric = norm_nelbo.detach().numpy().mean()
    return final_metric, all_metrics

def get_unsup_results(train_loader, train_data_dict, valid_loader, test_data_dict, device, model, epochs, sigmoid, ppmi=False, chf=False, fname=None, search='nelbo',anneal=False):
    """
    We are looking for the LOWEST metric over all the parameter searches. Make sure metrics are tuned accordingly!
    """
    hp = {}
    best_perf, best_mse, best_ari, best_swaps, best_pear = np.inf, np.inf, np.inf, np.inf, np.inf
    best_config = None
    all_results = {}
    
    MODEL = None
    if model == 'sublign_mog':
        MODEL = Sublign_MoG
    else:
        NotImplemented()
    print (MODEL)

#     if model =='sublign_mog':
    for K in [1,2,3]:
        for use_MAP in [True]:
            for anneal in [True, False]:
                for C in [0.,0.01]:
                    for ds in [5,20]:
                        for dim_h in [100,200]:
                            for dim_rnn in [200,500]:
                                for reg_type in ['l2']:
                                    for lr in [1e-3, 1e-2, 1e-1]:
                                        hp['K']              = K
                                        hp['use_MAP']        = use_MAP
                                        hp['C']              = C
                                        hp['dim_hidden']     = dim_h
                                        hp['dim_stochastic'] = ds
                                        hp['dim_rnn']        = dim_rnn
                                        hp['reg_type']       = reg_type
                                        hp['lr']             = lr
                                        hp['sigmoid']        = sigmoid
                                        hp['dim_biomarkers'] = 3 if sigmoid else 1
                                        if ppmi:
                                            hp['dim_biomarkers'] = 4
                                        if chf:
                                            hp['dim_biomarkers'] = 12

#                                         try:
                                        perf, other_metrics = run_model(MODEL, hp, lr, train_loader, train_data_dict, valid_loader, test_data_dict, device, epochs = epochs, ppmi=ppmi, eval_freq=25, fname=fname, chf=False, search=search, anneal=anneal)
#                                         print('NELBO: %.3f' % perf, (anneal, K, C, use_MAP, dim_h, ds, dim_rnn, reg_type, lr))
        #                                     print('MSE %.3f, ARI: %.3f' % (all_metrics['mse'], all_metrics['ari']), (C, b_vae, dim_h, ds, dim_rnn, reg_type, lr))
        #                                     except:
        #                                         perf, all_metrics, mse, ari, swaps, pear = 0., None, 0., 0., 0., 0.
        #                                         print('ERROR:', (C, b_vae, dim_h, ds, dim_rnn, reg_type, lr))
                                        all_results[(anneal, K, C, use_MAP, dim_h, ds, dim_rnn, reg_type, lr)] = (perf, other_metrics)
                                        if perf<best_perf:
                                            best_perf = perf; best_config = (anneal, K, C, use_MAP, dim_h, ds, dim_rnn, reg_type, lr)
            return best_perf, best_config, all_results

def run_cv(model = 'sublign', dataset = 'sigmoid', epochs=10, data_num=1, ppmi=False, chf=False, search='nll', anneal=True):
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
        fname = 'runs/data%d_mog_hptune.pt'
    
    # run hyperparamter search
    best_perf, best_config, all_results = get_unsup_results(train_loader, train_data_dict, valid_loader, valid_dict, device, model, epochs=epochs, sigmoid=use_sigmoid, ppmi=ppmi, chf=chf,fname=fname, anneal=anneal)

    print('Best ARI: %.3f' % best_perf)
#     print('MSE: %.3f, ARI: %.3f, Swaps: %.3f, Pear: %.3f' % (best_mse, best_ari, best_swaps, best_pear))
#     print ('Done fold ',fold)
    
    print ('Saving...')

#     anneal_str = 'w_anneal' if anneal else 'wo_anneal'
    if ppmi:
        fname = 'runs/'+model+'_ppmi.pkl'
    elif chf:
        fname = 'runs/'+model+'_chf.pkl'
    else:
        fname = 'runs/'+model+'_'+str(data_num) + '_' + '0602' + '_'+ str(epochs)+ '.pkl'
    
    with open(fname,'wb') as f:
        print('dumped pickle')
        pickle.dump(all_results, f)
        
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument('--epochs', action='store', type=int, default=1000, help="Number of epochs")
    parser.add_argument('--data_num', action='store', type=int, default=1, help="Data scenario number")
    parser.add_argument('--ppmi', action='store_true', help="Hyperparam search on PPMI data")
    parser.add_argument('--chf', action='store_true', help="Hyperparam search on CHF data")
    
    parser.add_argument('--kmeans', action='store_true', help="Hyperparam search based on kmeans ARI")
    parser.add_argument('--nelbo', action='store_true', help="Hyperparam search based on NELBO")
    parser.add_argument('--nll', action='store_true', help="Hyperparam search based on NLL")
    parser.add_argument('--cheat', action='store_true', help="Hyperparam search based on cheating")
    parser.add_argument('--anneal', action='store_true', help="Hyperparam search based on cheating")
    
    args = parser.parse_args()
        
    if args.cheat:
        search = 'cheat'
    elif args.nelbo:
        search = 'nelbo'
    else:
        search = 'mse'
        
    run_cv(model = 'sublign_mog', dataset='sigmoid', epochs=args.epochs, data_num=args.data_num, ppmi=args.ppmi, chf=args.chf, search=search, anneal=args.anneal)