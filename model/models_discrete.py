import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyro.distributions import MultivariateNormal, Normal, Independent

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score

import scipy
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

import sys
sys.path.append('../model/')
from utils import check_has_missing, quad_function, convert_XY_pack_pad

sys.path.append('../evaluation/')
from eval_utils import get_cluster_swap_metric, get_cluster_pear_metric

sys.path.append('../plot/')
from plot_utils import plot_latent_labels, plot_delta_comp


sys.path.append('../data/')
from load import sigmoid, quadratic, load_data_format, parkinsons
from load import chf as load_chf
from data_utils import parse_data
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
 'axes.labelsize': 'x-large',
 'axes.titlesize':'x-large',
 'xtick.labelsize':'x-large',
 'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

from matplotlib import pyplot as plt

class Model(nn.Module):
    def __init__(self):
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        super(Model, self).__init__()
        
    def forward(self,**kwargs):
        raise ValueError('Should be overriden')
    
    def get_masks(self, M):
        m_t    = ((torch.flip(torch.cumsum(torch.flip(M.sum(-1), (1,)), 1), (1,))>1.)*1)
        m_g_t  = (m_t.sum(-1)>1)*1.
        lens   = m_t.sum(-1)
        return m_t, m_g_t, lens
    
    def masked_gaussian_nll_3d(self, x, mu, std):
        nll        = 0.5*np.log(2*np.pi) + torch.log(std)+((mu-x)**2)/(2*std**2)
        masked_nll = nll
        return masked_nll
    
    def apply_reg(self, p, reg_type='l2'):
        if reg_type == 'l1':
            return torch.sum(torch.abs(p))
        elif reg_type=='l2':
            return torch.sum(p.pow(2))
        else:
            raise ValueError('bad reg')
            
    
    def fit(self, train_loader, valid_loader, epochs, lr, eval_freq=1, print_freq=1000, anneal = False, fname = None, verbose=False, plot_debug=False, epoch_debug=False):
        if verbose:
            eval_freq = 50
            
        opt = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-3)
        best_loss, best_ep = 100000, -1
        training_loss   = list()
        training_epochs = list()
        
        testing_loss    = list()
        testing_epochs  = list()
        
        if fname is not None: 
            logging.basicConfig(
                filename=fname[:-4]+'_loss.log', filemode='w',
                format='%(asctime)s - %(levelname)s \t %(message)s',
                level=logging.INFO)
        
        for epoch in range(1, epochs+1):
            self.train()
            batch_loss      = 0
            test_batch_loss = 0
            idx        = 0
            test_idx   = 0
            
            for data_tuples in train_loader:
                opt.zero_grad()
                
                loss = self.forward(*data_tuples)
#                 loss = loss.item()
            
                if epoch_debug:
                    training_loss.append(loss)
                
                loss.backward()
                opt.step()
                idx +=1
                batch_loss += loss.item()
                
            cur_mse = batch_loss/float(idx)
            training_loss.append(cur_mse)
            
            if epoch%eval_freq==0:
                self.eval()
                eval_loss = self.forward(*valid_loader.dataset.tensors)
                eval_loss = eval_loss.item()
                
                testing_loss.append(eval_loss)
                
                if eval_loss<best_loss:
                    best_loss  = eval_loss; best_ep = epoch
                    if fname is not None:
                        torch.save(self.state_dict(), fname)
                        
                if epoch_debug:
                    test_nelbo.append(nelbo)
                    test_nll.append(nll)
                    test_kl.append(kl)
                    
                    
                    train_Y, train_S, train_X, train_M, train_T = train_loader.dataset.tensors
                    test_Y, test_S, test_X, test_M, test_T      = valid_loader.dataset.tensors

                    """
                    step 1: get z using mu not sampling
                    step 2: K-means cluster these z and save centers
                    step 3: return theta_k = g1(z_k) for K clusters
                    """
                    train_z, _   = self.get_mu(train_X,train_Y)
                    train_z      = train_z.detach().numpy()


                    # for different cluster algs, plot labels and true subtypes
                    K = 2
                    km = KMeans(n_clusters=K)
                    km.fit(train_z)
                    self.subtypes_km = km

                    test_z, kl = self.get_mu(test_X,test_Y)
                    test_theta = self.infer_functional_params(test_z)
                    best_delta = self.get_best_delta(test_X, test_Y, test_M, test_theta, kl)

                    test_z        = test_z.detach().numpy()
                    test_clusters = self.subtypes_km.predict(test_z)
                    true_clusters = [int(i) for i in np.squeeze(test_S)]

                    test_M     = torch.ones_like(test_X)
                    test_mse   = self.get_mse(test_X, test_Y, test_M, test_theta, best_delta)
                    test_ari   = adjusted_rand_score(test_clusters, true_clusters)

                    test_swaps = get_cluster_swap_metric(test_clusters, test_T[:,0,0].detach().numpy(), best_delta.detach().numpy())
                    test_pear  = get_cluster_pear_metric(test_clusters, test_T[:,0,0].detach().numpy(), best_delta.detach().numpy())

                    test_ari_vals.append(test_ari)
                    test_mse_vals.append(test_mse)
                    test_swaps_vals.append(test_swaps)
                    test_pear_vals.append(test_pear)

                    test_batch_loss += eval_loss.item()
                    test_idx += 1
                    testing_loss.append(test_batch_loss/float(test_idx))
                    
                    likelihood = self.imp_sampling(train_X, train_Y, imp_samples=50)
                    train_likelihood.append(likelihood)
                    
                    likelihood = self.imp_sampling(test_X, test_Y, imp_samples=50)
                    test_likelihood.append(likelihood)
                
                if plot_debug:
                    train_Y, train_S, train_X, train_M, train_T = train_loader.dataset.tensors
                    test_Y, test_S, test_X, test_M, test_T      = valid_loader.dataset.tensors
                    
                    train_z, _   = self.get_mu(train_X,train_Y)
                    train_z      = train_z.detach().numpy()

                    # for different cluster algs, plot labels and true subtypes
                    K = 2
                    km = KMeans(n_clusters=K)
                    km.fit(train_z)
                    self.subtypes_km = km

                    test_z, kl = self.get_mu(test_X,test_Y)
                    test_theta = self.infer_functional_params(test_z)
                    best_delta = self.get_best_delta(test_X, test_Y, test_M, test_theta, kl)

                    test_z        = test_z.detach().numpy()
                    test_clusters = self.subtypes_km.predict(test_z)
                    true_clusters = [int(i) for i in np.squeeze(test_S)]

                    test_M     = torch.ones_like(test_X)
                    test_mse   = self.get_mse(test_X, test_Y, test_M, test_theta, best_delta)
                    test_ari   = adjusted_rand_score(test_clusters, true_clusters)
                    
                    plot_latent_labels(test_z, test_S, 'plots/pngs/lr_%.3f_%03d_latent.png' % (lr, epoch), title='Epoch %d, ARI: %.3f' % (epoch, test_ari))

                    plot_delta_comp(test_T[:,0,0].detach().numpy(), best_delta.detach().numpy(), 'plots/pngs/lr_%.3f_%03d_delta.png' % (lr, epoch), title='Epoch %d, Pear: %.3f' % (epoch, test_pear))
                    
                self.train()
                    
        self.best_loss = best_loss
        self.best_ep   = best_ep
        
#         if fname is not None and epochs > eval_freq:
#             print('loaded state_dict. loss: %.4f (ep %d)' % (best_loss, best_ep))
#             self.load_state_dict(torch.load(fname))
#             self.eval()

        self.training_loss = training_loss
        self.testing_loss  = testing_loss
        
        if plot_debug:
            import os
            import imageio

            png_dir = 'plots/pngs/'
            kargs = {'duration': 0.3}

            images = []
            for file_name in sorted(os.listdir(png_dir)):
                if file_name.endswith('_latent.png'):
                    file_path = os.path.join(png_dir, file_name)
                    images.append(imageio.imread(file_path))
            imageio.mimsave('plots/data%d_latent_%.3f.gif' % (self.data_num, lr), images, **kargs)

            images = []
            for file_name in sorted(os.listdir(png_dir)):
                if file_name.endswith('_delta.png'):
                    file_path = os.path.join(png_dir, file_name)
                    images.append(imageio.imread(file_path))
            imageio.mimsave('plots/data%d_delta_%.3f.gif' % (self.data_num, lr), images, **kargs)

            # delete everything when you're done
            for file_name in os.listdir(png_dir):
                root = os.getcwd()
                complete_fname = os.path.join(root, png_dir+file_name)
                if not os.path.isdir(complete_fname):
                    os.unlink(complete_fname)
            
        if epoch_debug:
            import pickle
            f = open('data%d_results_lr%.3f.pk' % (self.data_num, lr), 'wb')
            results = {'epochs': epochs, 
                       'eval_freq': eval_freq, 
                       'ari': test_ari_vals,
                       'mse': test_mse_vals,
                       'swaps': test_swaps_vals,
                       'pear': test_pear_vals,
                       'train_likelihood': train_likelihood,
                       'test_likelihood': test_likelihood,
                       'train_loss': training_loss,
                       'test_loss': testing_loss,
                       'best_nelbo': best_nelbo,
                       'best_nll': best_nll,
                       'best_kl': best_kl,
                       'best_ep': best_ep,
                       'train_nelbo': train_nelbo,
                        'train_nll': train_nll,
                        'train_kl': train_kl,
                       'test_nelbo': test_nelbo,
                        'test_nll': test_nll,
                        'test_kl': test_kl,
                       'train_M_sum': train_M.sum(),
                       'test_M_sum': test_M.sum()
                      }
            pickle.dump(results, f)
            f.close()
        
        return best_loss, best_ep
    
class SublignDiscrete(Model):
    def __init__(self, C=0.0, dim_biomarkers=3, 
                 reg_type = 'l2', sigmoid=True, learn_time=True, auto_delta=True, max_delta=10.,
                plot_debug=False, epoch_debug=False, beta=0.001, device='cpu', K=2,fix_z=False):
        """
        note no lr here. lr is in fit.
        """
        super(SublignDiscrete, self).__init__()
        self.n_biomarkers   = dim_biomarkers
        self.C              = C
        self.reg_type       = reg_type
        
        self.fix_z          = fix_z
        self.sigmoid        = sigmoid
        self.dim_K          = K
        self.dim_delta      = 50
        self.delta_values   = torch.arange(0,max_delta,max_delta/self.dim_delta)
        
#         self.dz_features    = self.dim_stochastic
#         rnn_input_size      = self.n_biomarkers + 1
        
#         self.subtypes_km = None
#         self.rnn       = nn.RNN(rnn_input_size, self.dim_rnn, 1, batch_first = True)
#         self.enc_h_mu  = nn.Linear(self.dim_rnn, self.dim_stochastic)
#         self.enc_h_sig = nn.Linear(self.dim_rnn, self.dim_stochastic)

        self.z_prob     = torch.ones(self.dim_K) / self.dim_K
        self.delta_prob = torch.ones(self.dim_delta) / self.dim_delta
        
        # initialize functions theta = g1(z)
        
        if self.sigmoid:
            self.theta_param = torch.nn.Parameter(torch.rand(self.n_biomarkers,self.dim_K,2))
            self.kappa       = torch.nn.Sigmoid()
        else:
            self.theta_param = torch.nn.Parameter(torch.rand(self.n_biomarkers,self.dim_K,3))
            self.kappa       = lambda x: x
        
        if device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
            
        self.debug  = {}
        self.beta   = beta
        self.data_num = 1
    
    def compute_log_likelihood(self, X,Y):
        N_patients, N_visits, N_biomarkers = Y.shape
        
        
        p_y_given_x_delta_z = torch.ones((N_patients, N_visits, N_biomarkers,self.dim_K, self.dim_delta))
        
        
        delta_repeat = self.delta_values[None, None, None, None,:].repeat(N_patients, N_visits, N_biomarkers,self.dim_K, 1)
        X_repeat     = X[:, :, :, None,None].repeat(1,1, N_biomarkers,self.dim_K, self.dim_delta)

        X_plus_delta = X_repeat + delta_repeat

        if self.sigmoid:
            theta_repeat0 = self.theta_param[None,None,:,:,0,None].repeat(N_patients, N_visits, 1,1,self.dim_delta)
            theta_repeat1 = self.theta_param[None,None,:,:,1,None].repeat(N_patients, N_visits, 1,1,self.dim_delta)
            
            inside_repeat = theta_repeat1 * X_plus_delta + theta_repeat0
        else:
            theta_repeat0 = self.theta_param[None,None,:,:,0,None].repeat(N_patients, N_visits, 1,1,self.dim_delta)
            theta_repeat1 = self.theta_param[None,None,:,:,1,None].repeat(N_patients, N_visits, 1,1,self.dim_delta)
            theta_repeat2 = self.theta_param[None,None,:,:,2,None].repeat(N_patients, N_visits, 1,1,self.dim_delta)
            
            inside_repeat = theta_repeat2 * X_plus_delta * X_plus_delta + theta_repeat1 * X_plus_delta + theta_repeat0
        
        mu       = self.kappa(inside_repeat)
        Y_repeat = Y[:, :, :, None,None].repeat(1,1, 1,self.dim_K, self.dim_delta)
        
        p_y_given_x_delta_z = torch.exp(-(Y_repeat - mu)**2 / 2) / (np.sqrt(2*3.141592))
          
        z_prob_rep     = self.z_prob[None, None, None,:,None].repeat(N_patients, N_visits,N_biomarkers,1,self.dim_delta)
        delta_prob_rep = self.delta_prob[None, None, None,None,:].repeat(N_patients, N_visits,N_biomarkers,self.dim_K,1)
        
        log_likelihood = p_y_given_x_delta_z * z_prob_rep * delta_prob_rep
        
        return log_likelihood
    
    def compute_log_likelihood_fix_z(self, X,Y,S):
        N_patients, N_visits, N_biomarkers = Y.shape

        # size: N_patients, self.n_biomarkers, 3 = P
        theta_patient_hardcode = self.theta_param[:,S[:,0].type(torch.LongTensor),:]
        
        theta_patient_hardcode = theta_patient_hardcode.permute([1,0,2])
        
        p_y_given_x_delta = torch.ones((N_patients, N_visits, N_biomarkers,self.dim_delta))
        
        delta_repeat = self.delta_values[None, None, None,:].repeat(N_patients, N_visits, N_biomarkers,1)
        X_repeat     = X[:, :,:,None].repeat(1,1, N_biomarkers, self.dim_delta)

        X_plus_delta = X_repeat + delta_repeat

        if self.sigmoid:
            theta_repeat0 = theta_patient_hardcode[:,None,:,0,None].repeat(1, N_visits, 1,self.dim_delta)
            theta_repeat1 = theta_patient_hardcode[:,None,:,1,None].repeat(1, N_visits, 1,self.dim_delta)
            
            inside_repeat = theta_repeat1 * X_plus_delta + theta_repeat0
        else:
            theta_repeat0 = theta_patient_hardcode[:,None,:,0,None].repeat(1, N_visits, 1,self.dim_delta)
            theta_repeat1 = theta_patient_hardcode[:,None,:,1,None].repeat(1, N_visits, 1,self.dim_delta)
            theta_repeat2 = theta_patient_hardcode[:,None,:,2,None].repeat(1, N_visits, 1,self.dim_delta)
            
            inside_repeat = theta_repeat2 * X_plus_delta * X_plus_delta + theta_repeat1 * X_plus_delta + theta_repeat0
        
        mu       = self.kappa(inside_repeat)
        Y_repeat = Y[:, :, :, None].repeat(1,1, 1,self.dim_delta)
        
        p_y_given_x_delta_z = torch.exp(-(Y_repeat - mu)**2 / 2) / (np.sqrt(2*3.141592))
          
#         z_prob_rep     = self.z_prob[None, None, None,None].repeat(N_patients, N_visits,N_biomarkers,self.dim_delta)
        delta_prob_rep = self.delta_prob[None, None, None,:].repeat(N_patients, N_visits,N_biomarkers,1)
        
        log_likelihood = p_y_given_x_delta_z * delta_prob_rep
        
        return log_likelihood
    
    def get_loss(self, X,Y,S):
        """
        input:
         - X (N_patients, N_visits, 1)
         - Y (N_patients, N_visits, N_biomarkers)
         - theta (N_patients, N_biomarkers each component) 
         - delta (N_patients)
        
        output:
         - yhat (N_patients, N_visits, N_biomarkers)
         
        step 1: convert everything to size N_patients, N_visits, N_biomarkers
        step 2: calculate loss yhat = f(x+delta; theta) 
        """
        
        if self.fix_z:
            log_likelihood = self.compute_log_likelihood_fix_z(X,Y,S)
        else:
            log_likelihood = self.compute_log_likelihood(X,Y)
        
        # clip values to 1e-10 to avoid nans
        log_likelihood = torch.clamp(log_likelihood,min=1e-10)
        
        if self.fix_z:
            loss = log_likelihood.sum(dim=[3])
        else:
            loss = log_likelihood.sum(dim=[3,4])
        loss = torch.log(loss)
        loss = loss.sum()
        
        
        return loss
    
    def forward(self, Y, S, X, M, T, anneal = 1.):
        if type(M) == np.ndarray:
            X = torch.tensor(X).to(self.device)
            Y = torch.tensor(Y).to(self.device)
            S = torch.tensor(S).to(self.device)
        
        loss = self.get_loss(X,Y,S)        
        return loss
    
    def get_delta(self, X,Y):
        """
        Returns delta probabilities for given X,Y values
        """
        if type(X) == np.ndarray:
            X = torch.tensor(X).to(self.device)
            Y = torch.tensor(Y).to(self.device)
        
        log_likelihood = self.compute_log_likelihood(X,Y)
        
        numerator   = log_likelihood.sum(dim=3)
        denominator = log_likelihood.sum(dim=[3,4])
        
        denominator = denominator[:,:,:,None].repeat(1,1,1,self.dim_delta)
        
        p_delta_given_x_y = torch.div(numerator,denominator)
        
        log_prob   = torch.log(p_delta_given_x_y)
        delta_prob = log_prob.sum(dim=[1,2])
        
        best_delta_idx = torch.argmax(delta_prob, dim=1)
        
        best_delta = np.ones(len(best_delta_idx))
        for idx, i in enumerate(best_delta_idx):
            best_delta[idx] = self.delta_values[i]
        return best_delta
        
    
    def get_z(self, X,Y):
        """
        Returns z likelihoods for given X,Y observations
        """
        if type(X) == np.ndarray:
            X = torch.tensor(X).to(self.device)
            Y = torch.tensor(Y).to(self.device)
        
        log_likelihood = self.compute_log_likelihood(X,Y)
        
        numerator   = log_likelihood.sum(dim=4)
        denominator = log_likelihood.sum(dim=[3,4])
        
        denominator = denominator[:,:,:,None].repeat(1,1,1,self.dim_K)
        
        p_z_given_x_y = torch.div(numerator,denominator)
        
        log_prob   = torch.log(p_z_given_x_y)
        z_prob     = log_prob.sum(dim=[1,2])
        
        best_z = torch.argmax(z_prob, dim=1) 
        return best_z
    
        
    def score(self, train_data_dict, test_data_dict, K=2):
        """
        step 1: get delta
        step 2: get subtype assignments
        step 3: get performance metrics
        """
        
        for col in ['Y_collect', 'obs_t_collect', 's_collect', 't_collect']:
            if col not in test_data_dict:
                print('ERROR: %s not in test_data_dict' % col)
                return 
        
#         cent_lst = self.get_subtypes(train_data_dict['obs_t_collect'], train_data_dict['Y_collect'], K=K)
        
        test_X = torch.tensor(test_data_dict['obs_t_collect']).to(self.device)
        test_Y = torch.tensor(test_data_dict['Y_collect']).to(self.device)
        test_M = torch.tensor(test_data_dict['mask_collect']).to(self.device)
        
        test_clusters   = self.get_z(test_X,test_Y)
        test_delta      = self.get_delta(test_X,test_Y)
        
        true_clusters = [int(i) for i in np.squeeze(test_data_dict['s_collect'])]
        
        test_ari   = adjusted_rand_score(test_clusters, true_clusters)

        test_swaps = get_cluster_swap_metric(test_clusters, test_data_dict['t_collect'][:,0,0], test_delta)
        test_pear  = get_cluster_pear_metric(test_clusters, test_data_dict['t_collect'][:,0,0], test_delta)
        
        results = {
            'ari': test_ari,
            'swaps': test_swaps,
            'pear': test_pear
        }
        
        return results
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument('--epochs', action='store', type=int, default=1600, help="Number of epochs")
    parser.add_argument('--data_num', action='store', type=int, default=1, help="Data scenario number")
    parser.add_argument('--K', action='store', type=int, default=2, help="Number of assumed subtypes")
    parser.add_argument('--lr', action='store', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--fix_z', action='store_true', help='Fix subtype to correct theta')
    
    args = parser.parse_args()
    
    data_num = args.data_num
    epochs   = args.epochs
    K        = args.K
    lr       = args.lr
    
    use_sigmoid = data_num < 3
    data        = load_data_format(data_num, 0)
    train_loader, train_data_dict, valid_loader, valid_data_dict, test_loader, test_data_dict, valid_pid, test_pid, unique_pid = parse_data(data.values, max_visits=4, test_per=0.2, valid_per=0.2, shuffle=False)

    model = SublignDiscrete(K=K, fix_z=args.fix_z)
    device  = torch.device('cpu')
    
    model.to(device)
    
    model.fit(train_loader, valid_loader, epochs, lr, eval_freq=25, fname='discrete%d.pt' % data_num)
    results = model.score(train_data_dict,valid_data_dict, K=K)
    
    sublign_ari = results['ari']
    sublign_swaps = results['swaps']
    sublign_pear = results['pear']
    
    print(results)
    
    from collections import Counter
    
    REDACTED = model.get_z(train_data_dict['obs_t_collect'], train_data_dict['Y_collect']).detach().numpy()
    c = Counter(REDACTED)
    print(c)

    REDACTED = model.get_z(test_data_dict['obs_t_collect'], test_data_dict['Y_collect']).detach().numpy()
    c = Counter(REDACTED)
    print(c)
    
    REDACTED = model.get_delta(train_data_dict['obs_t_collect'], train_data_dict['Y_collect'])
    c = Counter(REDACTED)
    print(c)

    plt.figure()
    plt.plot(range(epochs), np.array(model.training_loss) / 600,label='Train')
    plt.plot(np.arange(1,epochs,25), np.array(model.testing_loss) / 200, label='Test')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/loss_data%d.pdf' % data_num, bbox_inches='tight')

#     import pdb; pdb.set_trace()
    
    