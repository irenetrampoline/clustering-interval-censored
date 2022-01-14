import argparse
import numpy as np
import os
import sys
sys.path.append('../data')
sys.path.append('../plot')

import torch

from load import sigmoid, quadratic, chf, parkinsons, load_data_format
from data_utils import parse_data, change_missing
from plot_utils import plot_subtypes, plot_latent

from models import Sublign

def get_hyperparameters(data_format_num):
#     if data_format_num < 3:
#         anneal, b_vae, C, ds, dh, drnn, reg_type, lr = False, 0.001, 0., 10, 20, 50, 'l1', 0.01
    if data_format_num == 3:
        # failing on hpsearch
        anneal, C, b_vae, dh, ds, drnn, reg_type, lr = False, 0.01, 0.0, 100, 20, 200, 'l2', 0.001
#     if data_format_num == 5 or data_format_num == 3:
#         anneal, b_vae, C, ds, dh, drnn, reg_type, lr = False, 0.001, 0.01, 20, 20, 100, 'l2', 0.01
    if data_format_num == 1:
        # best by hpsearch: (True, 0.001, 0.0, 200, 5, 200, 'l2', 0.001)
        anneal, b_vae, C, ds, dh, drnn, reg_type, lr = True, 0.001, 0.0, 5, 200, 200, 'l2', 0.001
    if data_format_num == 3:
#         anneal, b_vae, C, ds, dh, drnn, reg_type, lr = True, 0.0, 0.0, 100, 20, 200, 'l2', 0.01
        anneal, b_vae, C, ds, dh, drnn, reg_type, lr = True, 1.0, 0.01, 100, 5, 200, 'l2', 0.01 # cheat
        anneal, b_vae, C, ds, dh, drnn, reg_type, lr = False, 0.01, 0.0, 100, 20, 200, 'l2', 0.001 # cheat 2
        
    if data_format_num == 4:
#         anneal, b_vae, C, ds, dh, drnn, reg_type, lr = False, 0.01, 0.0, 100, 20, 200, 'l2', 0.01 # cheat
        anneal, b_vae, C, ds, dh, drnn, reg_type, lr = False, 1.0, 0.01, 200, 20, 200, 'l2', 0.1
    if data_format_num == 5:
        anneal, b_vae, C, ds, dh, drnn, reg_type, lr = False, 0.01, 0.0, 200, 20, 200, 'l2', 0.01
    if data_format_num == 6:
        anneal, b_vae, C, ds, dh, drnn, reg_type, lr = True, 0.01, 0.0, 200, 20, 200, 'l2', 0.01
    if data_format_num == 7:
        anneal, b_vae, C, ds, dh, drnn, reg_type, lr = False, 0.01, 0.0, 100, 20, 200, 'l2', 0.001
    if data_format_num == 8:
        anneal, b_vae, C, ds, dh, drnn, reg_type, lr = True, 0.01, 0.01, 100, 20, 200, 'l2', 0.01

    anneal, b_vae, C, ds, dh, drnn, reg_type, lr = False, 1., 0., 10, 20, 50, 'l1', 1e-2
        
        # best from prev  : False, 0.001, 0.0, 10, 20, 50, 'l1', 0.1
#         anneal, b_vae, C, ds, dh, drnn, reg_type, lr = False, 0.001, 0.0, 10, 20, 50, 'l1', 0.1
    return anneal, b_vae, C, ds, dh, drnn, reg_type, lr
def get_hyperparameters_ppmi():
    b_vae, C, ds, dh, drnn, reg_type, lr = 0.01, 0., 10, 10, 20, 'l1', 0.1
    return b_vae, C, ds, dh, drnn, reg_type, lr

def get_hyperparameters_chf(version=0):
    # original, results in paper are from this version
    if version == 0:
        ds, dh, drnn, reg_type, C, b_vae, lr, epochs, learn_time = 10, 20, 50,'l1', 0.0, 0.001, 0.01, 1000, False
    elif version == 1:
        ds, dh, drnn, reg_type, C, b_vae, lr, epochs, learn_time = 10,200,200,'l1', 0.0, 0.001, 0.01, 1000, True
    elif version == 2:
        ds, dh, drnn, reg_type, C, b_vae, lr, epochs, learn_time = 10,200,200,'l1', 0.1, 0.001, 0.01, 1000, True
    elif version == 3:
        ds, dh, drnn, reg_type, C, b_vae, lr, epochs, learn_time = 10,200,200,'l1', 0.0, 0.001, 0.1, 1000, True
    elif version == 4:
        ds, dh, drnn, reg_type, C, b_vae, lr, epochs, learn_time = 10,200,200,'l1', 0.0, 0.01, 0.1, 1000, True
    return ds, dh, drnn, reg_type, C, b_vae, lr, epochs, learn_time
        
def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--epochs', action='store', type=int, default=800, help="Number of epochs")
    parser.add_argument('--trials', action='store', type=int, default=1, help="Number of trials")
    parser.add_argument('--model_name', action='store', type=str, default='SubLign', help="Model name for Latex table making")
    parser.add_argument('--lr', action='store', type=float, default=None, help="Learning rate manual override")
    parser.add_argument('--b_vae', action='store', type=float, default=None, help="b-VAE val override")
    parser.add_argument('--C', action='store', type=float, default=None, help="C override")

    # datasets
    parser.add_argument('--data_num', action='store', type=int, help="Data Format Number")
    parser.add_argument('--chf', action='store_true', help="Use CHF dataset")
    parser.add_argument('--ppmi', action='store_true', help="Use PPMI dataset")

    # delta setup
    parser.add_argument('--max_delta', action='store', type=float, default=5., help="Maximum possible delta")
    parser.add_argument('--no_time', action='store_true', help="Learn time at all")

    # debugging
    parser.add_argument('--verbose', action='store_true', help="Plot everything")
    parser.add_argument('--cuda', action='store_true', help="Use GPU")
    parser.add_argument('--missing', action='store', type=float, default=0., help="What percent of data to make missing")
    parser.add_argument('--plot_debug', action='store_true', help="Make animated gif about alignment / clusterings over epochs")
    parser.add_argument('--epoch_debug', action='store_true', help="Save pickle about epoch differences over training")
    parser.add_argument('--aggressive', action='store', type=int, help="Learn time at all")
    parser.add_argument('--version', action='store', type=int, help="Choose hyp settings", default=0)
    # other experiments

    args = parser.parse_args()

    trial_results = np.zeros((args.trials, 4))
    data_format_num = args.data_num

    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
        
    print('device', device)
    
    print('data %d' % data_format_num)
    for trial_num in range(args.trials):
        # datasets
        if data_format_num is not None:
            max_visits      = 4 
            num_output_dims = 3 if data_format_num < 3 else 1
            use_sigmoid     = data_format_num < 3
            anneal, b_vae, C, d_s, d_h, d_rnn, reg_type, lr = get_hyperparameters(data_format_num)
            
            if args.lr is not None:
                print('Running with lr=%.3f' % args.lr)
                lr = args.lr
                
            if args.C is not None:
                print('Running with C=%.3f' % args.C)
                C = args.C
                
            if args.b_vae is not None:
                print('Running with b_vae=%.3f' % args.b_vae)
                b_vae = args.b_vae

            data = load_data_format(data_format_num, trial_num, cache=True)
            shuffle = False
        elif args.chf:
            ds, dh, drnn, reg_type, C, b_vae, lr, epochs, learn_time = get_hyperparameters_chf(version=args.version)
            data       = chf()
            max_visits = 38
            shuffle    = True
            num_output_dims = data.shape[1] - 4
        elif args.ppmi:
            b_vae, C, d_s, d_h, d_rnn, reg_type, lr = get_hyperparameters_ppmi()
            
            if args.lr is not None:
                print('Running with lr=%.3f' % args.lr)
                lr = args.lr
                
            if args.C is not None:
                print('Running with C=%.3f' % args.C)
                C = args.C
                
            if args.b_vae is not None:
                print('Running with b_vae=%.3f' % args.b_vae)
                b_vae = args.b_vae
                
            data       = parkinsons()
            max_visits = 17
            shuffle    = True
            num_output_dims = data.shape[1] - 4

        train_data_loader, train_data_dict, _, _, test_data_loader, test_data_dict, valid_pid, test_pid, unique_pid = parse_data(data.values, max_visits=max_visits, test_per=0.2, valid_per=0.2, shuffle=shuffle, device=device)
        if args.missing > 0.:
            train_data_loader, train_data_dict = change_missing(train_data_dict, args.missing)

        data_loader, collect_dict, unique_pid = parse_data(data.values, max_visits=max_visits, device=device)

        """
        best parmas found through hypertuning (cross_validation/hpsearch.py)
        # sigmoid: C (0.01), dim_h (20), ds (10 mid), dim_rnn (50 mid), reg_type (l1), lr (0.1)
        # quad: C (0.1), dim_h (50), ds (10), dim_rnn (100), reg_type (l1), lr (0.1)

        ppmi: (0.0, 10, 10, 50, 'l1', 0.1)
        """

        # dim_stochastic, dim_hidden, dim_rnn, C, dim_biomarkers=3, reg_type = 'l2', 
        if data_format_num is not None:
            model = Sublign(d_s, d_h, d_rnn, b_vae=b_vae, dim_biomarkers=num_output_dims, sigmoid=use_sigmoid, reg_type=reg_type, auto_delta=False, max_delta=args.max_delta, learn_time=(not args.no_time), device=device)
            if device == 'cuda':
                device_torch = torch.device('cuda')
                model.to(device_torch)
            model.fit(train_data_loader, test_data_loader, args.epochs, lr, verbose=args.verbose, fname='runs/data%d_trial%d.pt' % (data_format_num, trial_num), eval_freq=25, anneal=anneal)
        elif args.chf:
            args.verbose = False
            model = Sublign(ds, dh, drnn, dim_biomarkers=num_output_dims, sigmoid=True, reg_type=reg_type, C=C, auto_delta=False, max_delta=args.max_delta, learn_time=(not args.no_time and learn_time), device=device, b_vae=b_vae)
            if device == 'cuda':
                device_torch = torch.device('cuda')
                model.to(device_torch)
            model.fit(data_loader, data_loader, args.epochs, lr, verbose=args.verbose,fname='runs/chf_v%d_%d.pt' % (args.version, args.epochs),eval_freq=25)
            
            
            X = torch.tensor(collect_dict['Y_collect']).to(model.device)
            Y = torch.tensor(collect_dict['obs_t_collect']).to(model.device)
            M = torch.tensor(collect_dict['mask_collect']).to(model.device)
            
            (nelbo, nll, kl), norm_reg = model.forward(Y, None, X, M, None)
            nelbo, nll, kl, norm_reg = nelbo.item(), nll.item(), kl.item(), norm_reg.item()
            
            subtypes = model.get_subtypes_datadict(collect_dict, K=3)
            labels   = model.get_labels(collect_dict)
            deltas   = model.get_deltas(collect_dict)
            
            if args.cuda:
                deltas   = deltas.cpu().detach().numpy()
            else:
                deltas   = deltas.detach().numpy()
                
            import pickle
            results = {
                'labels':labels,
                'deltas': deltas,
                'subtypes': subtypes,
                'nelbo': nelbo,
                'nll': nll,
                'kl': kl,
                'norm_reg': norm_reg
            }
            pickle.dump(results, open('../clinical_runs/chf_v%d_%d.pk' % (args.version, args.epochs), 'wb')) 
            return

        elif args.ppmi:
            model = Sublign(d_s, d_h, d_rnn, b_vae=b_vae, C=C, dim_biomarkers=num_output_dims, sigmoid=True, reg_type=reg_type, auto_delta=True, max_delta=args.max_delta, learn_time=(not args.no_time))
            model.fit(train_data_loader, test_data_loader, args.epochs, lr=lr, verbose=args.verbose, fname='runs/ppmi.pt', eval_freq=25)
            results  = model.score(train_data_dict, test_data_dict, K=2)
            test_ari = results['ari']
            print('PPMI Test ARI: %.3f' % test_ari)
            
#             results  = model.score(train_data_dict, test_data_dict, K=2)
#             test_ari = results['ari']
#             print('PPMI Test ARI: %.3f' % test_ari)
            
            subtypes = model.get_subtypes_datadict(collect_dict)
            labels   = model.get_labels(collect_dict)
            deltas   = model.get_deltas(collect_dict)
            import pickle
            
            if args.cuda:
                subtypes = subtypes.cpu().detach().numpy()
                labels   = labels.cpu().detach().numpy()
                deltas   = deltas.cpu().detach().numpy()
            else:
                subtypes = subtypes.cpu().detach().numpy()
                labels   = labels.cpu().detach().numpy()
                deltas   = deltas.cpu().detach().numpy()
                
            pickle.dump((labels, deltas, subtypes), open('../clinical_runs/ppmi_icml.pk', 'wb')) 
            return


    #     subtypes = model.get_subtypes(train_data_dict['obs_t_collect'], train_data_dict['Y_collect'], K=2)
        train_results = model.score(train_data_dict, train_data_dict)
        test_results  = model.score(train_data_dict, test_data_dict)
        
        train_mse   = train_results['mse']
        train_ari   = train_results['ari']
        train_swaps = train_results['swaps']
        train_pear  = train_results['pear']

        mse   = test_results['mse']
        ari   = test_results['ari']
        swaps = test_results['swaps']
        pear  = test_results['pear']

    #     nelbo, nll, kl = model.get_loss(Y, S, X, M, anneal=1.)
    #     nelbo, nll, kl = nelbo.mean().detach().numpy(), nll.mean().detach().numpy(), kl.mean().detach().numpy()

    #     if args.verbose:
    #         plot_subtypes(subtypes, args.sigmoid, train_data_dict)
    #         plot_latent(model, test_data_dict)
        trial_results[trial_num] = [mse, ari, swaps, pear]

    if args.no_time:
        args.model_name = 'SubNoLign'
        
    if args.trials == 1:
        print('Train: %.3f, %.3f, %.3f, %.3f' % (train_mse, train_ari, train_swaps, train_pear))
        print('Test : %.3f, %.3f, %.3f, %.3f' % (mse, ari, swaps, pear))
    #     print('NELBO: %.3f, NLL: %.3f, KL: %.3f' % (nelbo, nll, kl))
    else:
        line_str = list()
        for i,j in zip(trial_results.mean(axis=0), trial_results.std(axis=0)): 
            line_str.append('%.3f $\\pm$ %.3f' % (i,j))
        print(' & '.join([args.model_name] + line_str) + '\\\\')

        if args.data_num:
            trials_fname = '%s_data%d_trials%d.txt' % (args.model_name, args.data_num, args.trials)
        else:
            trials_fname = '%s_ppmi_trials%d.txt' % (args.model_name, args.trials)
        if not os.path.exists(trials_fname):
            f = open(trials_fname, 'w')
        else:
            f = open(trials_fname, 'a')

        f.write(' & '.join([args.model_name] + line_str) + '\\\\' + '\n')
        f.close()

if __name__=='__main__':
    main()