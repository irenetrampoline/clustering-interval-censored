import argparse
import numpy as np
import pystan

import pickle as pk
from scipy.stats import norm

from sklearn.metrics import adjusted_rand_score

import sys
sys.path.append('../evaluation/')
from eval_utils import get_cluster_swap_metric, get_cluster_pear_metric


# implement gauss sliding
sliding_gauss = """
   functions{
   }
   data {
       int D;          // number of dimensions
       int K;          // number of mixtures
       int N;          // number of data
       int T;          // number of time intervals
       int M;          // number of start times
       int MplusT;     // literally is just M + T
       matrix[N,T] y; // data
       matrix<lower=0,upper=1>[N,T] isnull; // 0/1 if data point is null
       
       real sig1;      // variance of x for each dim (SMALL)
       real sig2;      // variance of theta_kt for each dim (BIG)      
       vector<lower=0,upper=1>[K] alpha;      // dirichlet prior for cluster dist pi
       vector<lower=0,upper=1>[M] beta;      // prior for start time distributions
   }
   parameters {
       simplex[K] pi;             // mixing proportions of clust
       simplex[M] z;             // start time probability dist (aka z)
       
       matrix<lower=-10, upper=10>[K,MplusT] theta;           // means of x_it
   }
   model {       
       for (k in 1:K) {
          for (mt in 1:MplusT) {
              theta[k,mt] ~ normal(0, sig2);
          }
       }
       
       for (n in 1:N) {
         for (t in 1:T) {
             if (isnull[n,t] == 0) {
                 matrix[K,M] ps;
                 for (k in 1:K) {
                    for (m in 1:M) {
                        ps[k,m] = log(pi[k]) + log(z[m]) + normal_lpdf(y[n,t] | theta[k,m + t-1], sig1);
                    }
                 }
                 target += log_sum_exp(ps);
            } 
         }
       }
   }
"""

# want two generated quantities: subtype for each patient P(K | X), stage of each patient P(M | X)

class SlidingGauss:
    def __init__(self, data_num, trial_num, ppmi=False, ppmi_data=None):
        self.data_num  = data_num
        self.trial_num = trial_num
        self.ppmi      = ppmi
        self.ppmi_data = ppmi_data
        
    def create_fake_data(self):
        N = 1000
        D = 1

        M = 10 # start points
        T = 20 # recorded time
        pi = [0.25, 0.05, 0.30, 0.40] # dist of clusters
        m = [0.01, 0.04, 0.15, 0.15, 0.1, 0.1, 0.15, 0.15, 0.14, 0.01] # dist of starts

        self.T = T
        self.M = M
        self.K = len(pi)
        self.N = N
        
        # hard code clusters to make meaningful
        # cluster 1: slow linear decline
        # cluster 2: uniform but higher
        # cluster 3: uniform but lower
        # cluster 4: sharp decline then flat
        clusters = np.zeros((4,30))
        clusters[0] = [80 - 2.*i for i in range(30)]
        clusters[1] = [50]*30
        clusters[2] = [15]*30
        clusters[3] = [80 - 10.*i for i in range(7)] + [10] * 23

        per_missing = 0.
        
        cluster_assign = np.random.choice(len(pi), p=pi, size=N)
        start_assign = np.random.choice(len(m), p=m, size=N)

        sig1 = 5 # narrow window for each data point from cluster mean
        sig2 = 35 # cluster mean can vary (not used for fake data)

        self.fake_x = np.zeros((N, T))

        for i in range(N):
            # which cluster?
            clust_num = cluster_assign[i]
            clust_means = clusters[clust_num]

            # start point
            start_i = start_assign[i]
            for j in range(T):
                # is data point present?
#                 if np.random.random() > 0.:
                if np.random.random() > per_missing:
                    cur_t = start_i + j
                    x_ij = np.random.normal(loc=clust_means[cur_t], scale=sig1)
                    self.fake_x[i][j] = abs(x_ij)

                else:
                    self.fake_x[i][j] = np.nan
        
        self.fake_data = {'D':D, 'K': 4,  'N': N, 'T': T, 'M': M, 'MplusT': M+T, 
                          'y': self.fake_x, 'isnull': np.isnan(self.fake_x).astype(int),
                          'sig1': sig1, 'sig2': sig2, 'alpha': [1/4.] * 4, 'beta': m
                         }
        
    def run_fake(self):
        self.create_fake_data()
        self.fake = True
        self.run_model()
    
    def compute_conditionals(self, X, la):
        N = X.shape[0]
        pk_x = np.zeros(N)
        pm_x = np.zeros(N)
        
        # TODO: find highest likelihood sample
        best_idx = np.argmax(la['lp__'])
        
        # pi is shape N_subtypes
        # z is shape N_timepoints
        # theta is shape N_subtypes, N_timepoints
        
        pi    = la['pi'][best_idx]
        z     = la['z'][best_idx]
        theta = la['theta'][best_idx]
        
        
        # p(k|x) = sum_k p(k) p(x|m,k)
        # p(m|x) = sum_m p(m) p(x|m,k)
        
        # Step 1: compute p(x|m,k) for all combos of m,k (m * k options)
        p_x_given_m_k = np.zeros((N, self.M, self.K))
#         X = self.data['y']
        
        # for each patient
        for n in range(N):
            # for each learned start time
            for m in range(self.M):
                # for each learned subtype
                for k in range(self.K):
                    p_x_given_m_k[n, m,k] = self.norm_pdf(n,m,k, X, theta)
                    
        
        # Step 2: sum over axis
        # p(k) -> pi, p(m) -> z
        
        pi_reshaped = np.tile(pi, (N, self.M,1))
        z_reshaped  = np.tile(z[None,:,None], (N, 1, self.K))
        pm_x = (p_x_given_m_k + np.log(pi_reshaped)).sum(axis=2)
        pk_x = (p_x_given_m_k + np.log(z_reshaped)).sum(axis=1)
        
        subtypes = np.argmax(pk_x,axis=1)
        stages   = np.argmax(pm_x,axis=1)
        return subtypes, stages

    def norm_pdf(self, n, m,k, X, theta):
        """
        n: which number datapoint in X
        m: what is the start stage
        k: what is the subtype assignment
        X: observed data
        theta: learned param
        """
        log_pdf_sum = 0.
        
        for t in range(self.T):
            if not np.isnan(X[n,t]):
                prob_x = norm.pdf(X[n,t], loc=theta[k,m+t], scale=5)
                log_pdf_sum += np.log(prob_x)
        return log_pdf_sum
        # starting at stage m, add log probabilities? Then take exp at the end
        
#         prob_x = norm.pdf(X[n], loc=theta[m,k], scale=5)
#         return prob_x
        
    def get_data(self, train_data_dict):
        # This assumes evenly measured time after start time, but our data is irregularly spaced
        
        D = train_data_dict['Y_collect'].shape[2]
        K = 2
        
        T = 10
        M = 10
        Y = None
        sig1 = 2
        sig2 = 5
        
        self.T = T
        self.M = M
        self.K = K
        
        N = train_data_dict['t_collect'].shape[0]
        self.N = N
        
        Y = self.get_X(train_data_dict)
        
        data = {'D':D, 'K': K,  'N': N, 'T': T, 'M': M, 'MplusT': M+T, 
                          'y': Y, 'isnull': np.isnan(Y).astype(int),
                          'sig1': sig1, 'sig2': sig2, 'alpha': [1./K] * K, 'beta': [1./M] * M
                         }
        return data
        
    def run_model(self, train_data_dict, fake, iters=100):
        if fake:
            self.create_fake_data()
            data = self.fake_data
            self.data = data
#         elif self.ppmi:
#             data = self.get_data(self.ppmi_data)
#             self.data = data
        else:
            data = self.get_data(train_data_dict)
            self.data = data
            
        
        # TODO: load cached data
        # TODO: calculate p(x|m)
        
        sm = pystan.StanModel(model_code=sliding_gauss)
        fit = sm.sampling(data=data, iter=iters, chains=4, control={'adapt_delta': 2.})
        la = fit.extract(permuted=True)
        
        # save learned la
        
#         import pickle
#         f = open('huo_data%d_trial%d.pk' % (self.data_num, self.trial_num), 'wb')
#         pickle.dump(la, f)
#         f.close()
        
        self.la = la
        
    def get_X(self, train_data_dict):
        X_time = train_data_dict['t_collect']
        Y_obs  = train_data_dict['Y_collect']
        
        N = X_time.shape[0]
        T = self.T
        Y = np.empty((N,T))
        Y[:] = np.nan
        
        bins = np.arange(0,10,0.5)
        for n in range(N):
            bin_idx = np.digitize(X_time[n,:,0], bins)
            first_bin = bin_idx[0]
            valid_idx = [i for i in bin_idx if i < first_bin+10]
            for orig_visit, idx in enumerate(valid_idx):
                Y[n,idx-first_bin] = Y_obs[n,orig_visit,0]
        return Y
    
    def score(self, data_dict, test=False):
        if test:
            import pickle
            f = open('sublign_gauss_la.pk', 'rb')
            la = pickle.load(f)
            f.close()
        else:
            la = self.la
#         import pickle
#         la = pickle.load(open('example_la.pk', 'rb'))
        
        X = self.get_X(data_dict)
        subtypes, stages = self.compute_conditionals(X, la)
        
#         f = open('sublign_gauss_subtypes_stages.pk', 'wb')
#         pickle.dump((subtypes, stages),f)
#         f.close()
        
        gt_subtypes = data_dict['s_collect'].squeeze()
        gt_stages   = data_dict['t_collect'][:,0,:].squeeze()
        
        
        test_ari, test_swaps,test_pear  = evaluate_huopaniemi(subtypes, gt_subtypes, stages, gt_stages)

        print('Test ARI: %.3f' % test_ari)
        print('Test Swaps: %.3f' % test_swaps)
        print('Test Pear: %.3f' % test_pear) 
        return test_ari, test_swaps, test_pear

def evaluate_huopaniemi(test_subtypes, test_gt_subtypes, test_gt_stages, test_stages):
#     import pdb; pdb.set_trace()
    pear   = get_cluster_pear_metric(test_subtypes, test_gt_stages, test_stages)
    swaps  = get_cluster_swap_metric(test_subtypes, test_gt_stages, test_stages)    
    ari    = adjusted_rand_score(test_subtypes, test_gt_subtypes)
    return ari, swaps, pear
    
# km_mse, km_ari, km_swaps, km_pear, km_clusters, km_delta, km_params 
# = run_kmeans_baseline(train_data_dict, test_data_dict, use_sigmoid=use_sigmoid, full_output=True)
def run_huopaniemi_baseline(data_num, iters, trial_num, ppmi=False):
    import sys
    sys.path.append('../data/')
    from load import load_data_format
    from data_utils import parse_data

    if ppmi:
        from load import parkinsons
        data = parkinsons()
        max_visits = 17
    else:
        data = load_data_format(data_num, trial_num=trial_num, cache=True)
        max_visits = 4
    _, train_data_dict, _, test_data_dict, _, _ = parse_data(data.values, max_visits=max_visits, test_per=0.2)
    
    model = SlidingGauss(data_num, trial_num, ppmi)
    model.run_model(train_data_dict, fake=False, iters=iters)
    ari, swaps, pear = model.score(test_data_dict)
    return ari, swaps, pear
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--ppmi', action='store_true', help='Use PPMI data')
    parser.add_argument('--data_num', action='store', type=int, default=None, help='Run on synthetic SubLign data')
    parser.add_argument('--fake', action='store_true', help='Run on fake data from original paper')
    parser.add_argument('--iters', action='store', type=int, help='Number of iterations', default=100)
    
    args = parser.parse_args()
    
#     if args.data_num is None and not args.ppmi:
#         N_trials = 5

#         all_data_results = dict()

#         for data_num in [1,3,4,5,6,7,8]:
#             huo_trial_results = np.zeros((N_trials,3))
#             for trial_num in range(N_trials):
#                 try:
#                     ari, swaps, pear = run_huopaniemi_baseline(data_num=data_num, iters=args.iters, trial_num=trial_num)
#                 except:
#                     ari, swaps, pear = 0., 0., 0.
#                 huo_trial_results[trial_num] = [ari, swaps, pear]
#             all_data_results[data_num] = huo_trial_results

#         import pickle
#         f = open('huo_results.pk', 'wb')
#         pickle.dump(all_data_results, f)
#     else:
    ari, swaps, pear = run_huopaniemi_baseline(data_num=args.data_num, iters=args.iters, trial_num=1, ppmi=args.ppmi)
    print(ari)