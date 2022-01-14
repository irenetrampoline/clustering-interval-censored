import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import scipy.optimize as opt
import sys
sys.path.append('../evaluation/')
from eval_utils import get_cluster_swap_metric, get_cluster_pear_metric

class Dummy:
    pass

def sigmoid(x):
    return 1./(1+np.exp(-x))

def get_loss(params, X, Y, use_sigmoid):
    return

def get_ari(train_data_dict, test_data_dict, K=2):
    Y = train_data_dict['Y_collect']
    t = train_data_dict['obs_t_collect'].squeeze()
    N_patients, N_visits, N_dims = Y.shape
    
    test_Y = test_data_dict['Y_collect']
    test_t = test_data_dict['obs_t_collect'].squeeze()
    test_N_patients, test_N_visits, test_N_dims = test_Y.shape
    
    # Reshape data 
    Y = np.reshape(Y, (N_patients, N_visits * N_dims))
    test_Y = np.reshape(test_Y, (test_N_patients, test_N_visits*test_N_dims))
    
    km = KMeans(n_clusters=K).fit(Y)
    train_labels = km.predict(Y)
    test_labels = km.predict(test_Y)
    
    train_ari = adjusted_rand_score(train_labels, [int(i) for i in np.squeeze(train_data_dict['s_collect'])])
    test_ari  = adjusted_rand_score(test_labels, [int(i) for i in np.squeeze(test_data_dict['s_collect'])])
    print('Train ARI: %.3f' % train_ari)
    print('Test ARI: %.3f' % test_ari)
    
    return train_labels, test_labels, train_ari, test_ari, km

def get_real_solution_data4(train_data_dict, train_labels):
    res = Dummy()
    
    """
    subtype 1: y = 0.25 x^2 - 2.2 x + 5
    subtype 2: y = -2
    """
    
    X = train_data_dict['t_collect']
    Y = train_data_dict['Y_collect']
    S = train_data_dict['s_collect']
    
    # get all datapoints that are s=0 (by pred labels or true)
    s0_idx = np.where(train_labels==0)[0]
    s1_idx = np.where(train_labels==1)[0]
    
    Y_s0   = Y[s0_idx]
    Y_s1   = Y[s1_idx]
    
    # one of them is mostly flat (y = -2). which one?
    flat_const = -2
    if (flat_const - Y_s0.mean()) ** 2 > (flat_const - Y_s1.mean()) ** 2:
        # choose s1
        flat_idx  = 1
        curve_idx = 0
    else:
        # choose s0
        flat_idx  = 0
        curve_idx = 1
    
    a, b, c = np.zeros((1,2)), np.zeros((1,2)), np.zeros((1,2))
    
    a[0,flat_idx], b[0,flat_idx], c[0, flat_idx] = 0, 0, flat_const
    a[0,curve_idx], b[0,curve_idx], c[0, curve_idx] = 0.25, -2.2, 5
    
    delta = train_data_dict['t_collect'][:,0,0].flatten()
    
    x0  = np.concatenate([a.flatten(), b.flatten(), c.flatten(), delta.flatten()])
    
    res.x = x0
    return res

def get_yhat(params, x, t, use_sigmoid):
    x_plus_t = x+t
    
    if use_sigmoid:
        beta0, beta1 = params
        return sigmoid(beta0 + beta1 * x_plus_t)
    else:
        a, b, c  = params
        return a * x_plus_t * x_plus_t + b * x_plus_t + c
    
def compute_mse_labels(params, delta, X, Y, labels, use_sigmoid):
    """
    assume a, b, c, delta, X, Y are all in the right dimensions
    """
    if use_sigmoid:
        sig0, sig1 = params
    else:
        a,b,c = params
        
    N_patients, N_visits, N_dims = Y.shape
    all_mse = 0.
    
    for i in range(N_patients):
        for j in range(N_dims):
            subtype_i = labels[i]
            if use_sigmoid:
                sig0_i = sig0[j,subtype_i]
                sig1_i = sig1[j,subtype_i]
                params_i = (sig0_i, sig1_i)
            else:
                a_i       = a[j, subtype_i]
                b_i       = b[j, subtype_i]
                c_i       = c[j, subtype_i]
                params_i  = (a_i, b_i, c_i)
                
            Yhat    = get_yhat(params_i, X[i], delta[i], use_sigmoid)
            cur_mse = ((Yhat - Y[i,:,j]) **2).sum()
#             import pdb; pdb.set_trace()
            all_mse += cur_mse
            
    s0_idx = np.where(labels==0)[0]
    s1_idx = np.where(labels==1)[0]
    
    Y_s0   = Y[s0_idx]
    Y_s1   = Y[s1_idx]
    
    return all_mse / (N_patients * N_visits * N_dims)
    
def get_test_delta(params_best, test_t, test_Y, test_labels, use_sigmoid, true_test_delta=np.zeros(10)):
    N_test_patients = test_t.shape[0]
    
    def f_test(x0):
        if use_sigmoid:
            sig0, sig1 = params_best
        else:
            a,b,c = params_best

        test_N_patients, test_N_visits, test_N_dims = test_Y.shape
        all_mse = 0.

        for i in range(test_N_patients):
            for j in range(test_N_dims):
                subtype_i = test_labels[i]
                if use_sigmoid:
                    sig0_i = sig0[j,subtype_i]
                    sig1_i = sig1[j,subtype_i]
                    params_dim = (sig0_i, sig1_i)
                else:
                    a_i       = a[j, subtype_i]
                    b_i       = b[j, subtype_i]
                    c_i       = c[j, subtype_i]
                    params_dim = (a_i, b_i, c_i)
                Yhat = get_yhat(params_dim, test_t[i], x0[i], use_sigmoid)
                all_mse += ((Yhat - test_Y[i,:,j]) **2).sum()

        return all_mse
    
    delta0 = np.random.random(N_test_patients)
    
    if true_test_delta.sum() != 0.:
        print('Using the true test delta values')
        delta0 = true_test_delta
        
    res = opt.minimize(f_test, delta0, method='BFGS', options={'disp': False})
    return res.x

def run_kmeans_baseline(data_dict,test_data_dict, K=2, use_sigmoid=True, full_output=False, init_type='rand'):
    # K-means on raw data
    train_labels, test_labels, train_ari, test_ari, km = get_ari(data_dict, test_data_dict, K=2)
    
    Y = data_dict['Y_collect']
    t = data_dict['obs_t_collect'].squeeze()
    real_t = data_dict['t_collect'].squeeze()
    N_patients, N_visits, N_dims = Y.shape
    
    test_Y = test_data_dict['Y_collect']
    test_t = test_data_dict['obs_t_collect'].squeeze()
    test_real_t = test_data_dict['t_collect'].squeeze()
    true_test_delta = test_data_dict['t_collect'][:,0,:].squeeze()
    test_N_patients, test_N_visits, test_N_dims = test_Y.shape
    
    N_clusters = K
    
    def reshape_input(full_input):
        if use_sigmoid:
            sig0   = np.reshape(full_input[:N_dims*N_clusters],(N_dims, N_clusters))
            sig1   = np.reshape(full_input[N_dims*N_clusters:2*N_dims*N_clusters], (N_dims, N_clusters))
            delta  = full_input[2*N_dims*N_clusters:]
            params = (sig0, sig1)
        else:
            a      = np.reshape(full_input[:N_dims*N_clusters],(N_dims, N_clusters))
            b      = np.reshape(full_input[N_dims*N_clusters:2*N_dims*N_clusters], (N_dims, N_clusters))
            c      = np.reshape(full_input[2*N_dims*N_clusters:3*N_dims*N_clusters], (N_dims, N_clusters))
            delta  = full_input[3*N_dims*N_clusters:]
            params = (a, b, c)
        return params, delta
    
    def reshape_output(x):
        if use_sigmoid:
            sig0_best  = np.reshape(x[:N_dims*N_clusters],(N_dims, N_clusters))
            sig1_best  = np.reshape(x[N_dims*N_clusters:2*N_dims*N_clusters],(N_dims, N_clusters))
            delta_best = x[2*N_dims*N_clusters:]
            params_best = (sig0_best, sig1_best)
        else:
            a_best = np.reshape(x[:N_dims*N_clusters],(N_dims, N_clusters))
            b_best = np.reshape(x[N_dims*N_clusters:2*N_dims*N_clusters],(N_dims, N_clusters))
            c_best = np.reshape(x[2*N_dims*N_clusters:3*N_dims*N_clusters],(N_dims, N_clusters))
            
            delta_best  = x[3*N_dims*N_clusters:]
            params_best = (a_best, b_best, c_best)
            
        return params_best, delta_best
    
    def evaluate_kmeans_loss(x, t, Y, train_labels, use_sigmoid, true_test_delta=np.zeros(10), full_output=False):
        params_best, train_delta = reshape_output(x)
        
        train_mse  = compute_mse_labels(params_best, train_delta, t, Y, train_labels, use_sigmoid)
        test_delta = get_test_delta(params_best, test_t, test_Y, test_labels, use_sigmoid, true_test_delta=true_test_delta)
        test_mse   = compute_mse_labels(params_best, test_delta, test_t, test_Y, test_labels, use_sigmoid)
        pear       = get_cluster_pear_metric(test_labels, test_real_t[:,0], test_delta)
        swaps      = get_cluster_swap_metric(test_labels, test_real_t[:,0], test_delta)
        
        return test_mse, pear, swaps, test_delta, params_best
    
    def f_train(full_input):
        def get_mse_label(ti, delta_i, Yi, subtypei, params):
            """
            given a label for a specific person, compute mse
            """
            total = 0.
            
            if use_sigmoid:
                sig0, sig1 = params
            else:
                a, b, c = params 
            
            N_dims, N_clusters = params[0].shape
            
            for j in range(N_dims):
                if use_sigmoid:
                    params_dim = (sig0[j, subtypei], sig1[j, subtypei])
                else:
                    params_dim = (a[j, subtypei], b[j, subtypei], c[j, subtypei])
                yhat       = get_yhat(params_dim, ti, delta_i, use_sigmoid) 
                mse        = (yhat - Yi[:,j]) ** 2
            return mse.sum()

        """
        delta is offset
        t is observed X values
        """
        params, delta = reshape_input(full_input)
        
        total = 0.
        # for each patient in train data
        for i in range(N_patients):
            subtype_i = train_labels[i]
            mse       = get_mse_label(t[i], delta[i], Y[i], subtype_i, params)
            total += mse
        return total / (N_patients * N_dims * N_visits)
    
    if init_type == 'rand':
        delta = np.random.random(N_patients)  * 3
    elif init_type == 'zero':
        delta = np.zeros(N_patients)
    
    # delta = np.random.random(N_patients)
    
    if use_sigmoid:
        sig0 = np.zeros((N_dims, N_clusters))
        sig1 = np.zeros((N_dims, N_clusters))
        x0   = np.concatenate([sig0.flatten(), sig1.flatten(), delta.flatten()])
    else:
        a   = np.zeros((N_dims, N_clusters))
        b   = np.zeros((N_dims, N_clusters))
        c   = np.zeros((N_dims, N_clusters))
        x0  = np.concatenate([a.flatten(), b.flatten(), c.flatten(), delta.flatten()])

    if init_type == 'correct':
    # TODO: what if we give the right answers to initialize?
        res = get_real_solution_data4(data_dict, train_labels)
        x0  = res.x
    
#     res = get_real_solution_data4(data_dict, train_labels)
    
    res = opt.minimize(f_train, x0, method='BFGS',
                       options={'disp': False})

    test_mse, test_swaps, test_pear, test_delta, test_params = evaluate_kmeans_loss(res.x, t, Y, train_labels, use_sigmoid, true_test_delta=np.zeros(10), full_output=True)
    
    print('MSE: %.5f\n' % test_mse, 
          'ARI: %.3f\n' % test_ari,
          'Swaps: %.3f\n' % test_swaps,
          'Pearson: %.3f\n' % test_pear
         )
    
    if full_output:
        return test_mse, test_ari, test_swaps, test_pear, test_labels, test_delta, test_params
    else:
        return test_mse, test_ari, test_swaps, test_pear
    
if __name__=='__main__':
    import os.path
    import numpy as np
    import sys
    sys.path.append('../data/')
    sys.path.append('../model/')

    from load import load_data_format
    from data_utils import parse_data
    from models import Sublign
    
    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument('--trials', action='store', type=int, default=1, help="Number of trials")
    parser.add_argument('--data_num', action='store', type=int, default=1, help="Data Format Number")
    parser.add_argument('--init_type', action='store', type=str, default='rand', help="How to initialize values")
    parser.add_argument('--how_impute', action='store', type=str, default='mice', help="Options: mice and MRNN")
    args = parser.parse_args()
    
    trials          = args.trials
    data_format_num = args.data_num
    use_sigmoid     = (data_format_num == 1) or (data_format_num > 10)
    kmeans_trial_results    = np.zeros((trials, 4))
    
    max_visits = 4 if data_format_num < 10 else 17
    

    for trial_num in range(trials):
        data = load_data_format(data_format_num, trial_num)
        train_data_loader, train_data_dict, test_data_loader, test_data_dict, p_ids, full_p_ids = parse_data(data.values, max_visits=max_visits, test_per=0.2)
        
        if data_format_num == 11 or data_format_num == 12:
            X = train_data_dict['obs_t_collect']
            Y = train_data_dict['Y_collect']
            M = train_data_dict['mask_collect']

            X[X == -1000] = np.nan
            Y[Y == -1000] = np.nan

            sys.path.append('../model')
            from utils import interpolate

            if args.how_impute == 'mrnn':
                Y[np.isnan(Y)] = 0.
                X[np.isnan(X)] = 0.
                
#             import pdb; pdb.set_trace()
                
            Y_impute = interpolate(Y, m=M, t=X, how=args.how_impute)
            
            train_data_dict['Y_collect'] = Y_impute
            
            X = test_data_dict['obs_t_collect']
            Y = test_data_dict['Y_collect']
            M = test_data_dict['mask_collect']

            X[X == -1000] = np.nan
            Y[Y == -1000] = np.nan
            

            if args.how_impute == 'mrnn':
                Y[np.isnan(Y)] = 0.
                X[np.isnan(X)] = 0.
                
            Y_impute = interpolate(Y, m=M, t=X, how=args.how_impute)
            
            test_data_dict['Y_collect'] = Y_impute
    
        # sigmoid best params: C (0.01), dim_h (20), ds (10 mid), dim_rnn (50 mid), reg_type (l1), lr (0.1)
        
#         # run SubLign
#         model = Sublign(10, 20, 50, 0.01, 3, sigmoid=True, reg_type='l1', auto_delta=True, max_delta=10., learn_time=True)
#         model.fit(train_data_loader, test_data_loader, epochs, 0.01, verbose=False)
#         subtypes = model.get_subtypes(train_data_dict['obs_t_collect'], train_data_dict['Y_collect'], K=2)
#         sublign_mse, sublign_ari, sublign_swaps, sublign_pear = model.score(test_data_dict)
#         sublign_trial_results[trial_num] = [sublign_mse, sublign_ari, sublign_swaps, sublign_pear]
        
#         # run SubNoLign
#         model = Sublign(10, 20, 50, 0.01, 3, sigmoid=True, reg_type='l1', auto_delta=False, max_delta=10., learn_time=False)
#         model.fit(train_data_loader, test_data_loader, epochs, 0.01, verbose=False)
#         subtypes = model.get_subtypes(train_data_dict['obs_t_collect'], train_data_dict['Y_collect'], K=2)
#         subnolign_mse, subnolign_ari, subnolign_swaps, subnolign_pear = model.score(test_data_dict)
#         subnolign_trial_results[trial_num] = [subnolign_mse, subnolign_ari, subnolign_swaps, subnolign_pear]
        
        # run KMeans
        kmeans_mse, kmeans_ari, kmeans_swaps, kmeans_pear = run_kmeans_baseline(train_data_dict, test_data_dict, use_sigmoid=use_sigmoid, init_type=args.init_type)
#         kmeans_mse, kmeans_ari, kmeans_swaps, kmeans_pear = 0., 0., 0., 0.
        kmeans_trial_results[trial_num] = [kmeans_mse, kmeans_ari, kmeans_swaps, kmeans_pear]
    
    if trials == 1:
#         print('%s, %.3f, %.3f, %.3f, %.3f' % ('SubLign', sublign_mse, sublign_ari, sublign_swaps, sublign_pear))
#         print('%s, %.3f, %.3f, %.3f, %.3f' % ('SubNoLign',subnolign_mse, subnolign_ari, subnolign_swaps, subnolign_pear))
        print('%s, %.3f, %.3f, %.3f, %.3f' % ('KMeans+Loss',kmeans_mse, kmeans_ari, kmeans_swaps, kmeans_pear))
    else:
#         line_str = list()
#         for i,j in zip(sublign_trial_results.mean(axis=0), sublign_trial_results.std(axis=0)): 
#             line_str.append('%.3f $\\pm$ %.3f' % (i,j))
#         print(' & '.join(['SubLign'] + line_str) + '\\\\')
        
#         line_str = list()
#         for i,j in zip(subnolign_trial_results.mean(axis=0), subnolign_trial_results.std(axis=0)): 
#             line_str.append('%.3f $\\pm$ %.3f' % (i,j))
#         print(' & '.join(['SubNoLign'] + line_str) + '\\\\')
        
        line_str = list()
        for i,j in zip(kmeans_trial_results.mean(axis=0), kmeans_trial_results.std(axis=0)): 
            line_str.append('%.3f $\\pm$ %.3f' % (i,j))
        print(' & '.join(['KMeans+Loss'] + line_str) + '\\\\')
        
        trials_fname = 'runs/kmeans_data%d.txt' % data_format_num
        f = open(trials_fname, 'w')
        f.write(' & '.join(['KMeans+Loss'] + line_str) + '\\\\')
        f.close()