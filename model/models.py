import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


from pyro.distributions import MultivariateNormal, Normal, Independent

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score

import scipy
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

import sys
sys.path.append('/home/REDACTED/chf-github/model/')
from utils import check_has_missing, quad_function, convert_XY_pack_pad

sys.path.append('../evaluation/')
from eval_utils import get_cluster_swap_metric, get_cluster_pear_metric

sys.path.append('../plot/')
from plot_utils import plot_latent_labels, plot_delta_comp


import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
# 'figure.figsize': (10,6),
 'axes.labelsize': 'x-large',
 'axes.titlesize':'x-large',
 'xtick.labelsize':'x-large',
 'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

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
        best_nelbo, best_nll, best_kl, best_ep = 100000, 100000, 100000, -1
        training_loss   = list()
        training_epochs = list()
        
        testing_loss    = list()
        testing_epochs  = list()
        
        test_ari_vals  = list()
        test_mse_vals  = list()
        test_pear_vals = list()
        test_swaps_vals = list()
        
        
        train_nelbo = list()
        train_nll = list()
        train_kl = list()
        
        test_nelbo = list()
        test_nll = list()
        test_kl = list()
        
        train_likelihood = list()
        test_likelihood  = list()
        
        train_affinity_num_clusters = list()
        test_affinity_num_clusters  = list()
        
        if fname is not None: 
            logging.basicConfig(
                filename=fname[:-4]+'_loss.log', filemode='w',
                format='%(asctime)s - %(levelname)s \t %(message)s',
                level=logging.INFO)
        
        if anneal:
            anneal = 0.01
#             print ('With annealing starting at ',anneal)
        else:
#             print ('No annealing')
            anneal = 1.
    
        # TODO: consider caching the convert_XY_pad content because it's the bulk of the computation?
        """
        
        if check_has_missing(X) or check_has_missing(Y):
            has_missing = True
        else:
            has_missing = False
        
        XY = concat(X,Y)
        newXY, all_seq_lengths = convert_XY_pack_pad(XY)
        """
        
        Y, S, X, M, T = [i for i in train_loader][0]
        has_missing = False
        newXY       = None
        all_seq_lengths = None

        if check_has_missing(X) or check_has_missing(Y):
            has_missing = True
            XY = torch.cat([X,Y], axis=2)
            newXY, all_seq_lengths = convert_XY_pack_pad(XY, how=self.how_missing)
        else:
            has_missing = False
        

        # now validation
        val_Y, val_S, val_X, val_M, val_T = [i for i in valid_loader][0]
        val_has_missing = False
        val_newXY       = None
        val_all_seq_lengths = None
        if check_has_missing(val_X) or check_has_missing(val_Y):
            val_has_missing = True
            val_XY = torch.cat([val_X,val_Y], axis=2)
            val_newXY, val_all_seq_lengths = convert_XY_pack_pad(val_XY, how=self.how_missing)
        else:
            val_has_missing = False
        
        for epoch in range(1, epochs+1):
            anneal = min(1, epoch/(epochs*0.5))
            self.train()
            batch_loss      = 0
            test_batch_loss = 0
            idx        = 0
            test_idx   = 0
            
            for data_tuples in train_loader:
                opt.zero_grad()
#                 if epoch == 3:
                
                (nelbo, nll, kl), loss  = self.forward(*data_tuples, anneal = anneal,
                                                       has_missing=has_missing,XY=newXY, 
                                                       all_seq_lengths=all_seq_lengths)
                nelbo, nll, kl = nelbo.item(), nll.item(), kl.item()
            
                if epoch_debug:
                    train_nelbo.append(nelbo)
                    train_nll.append(nll)
                    train_kl.append(kl)
                
#                 from torch.autograd import grad
#                 grad(loss, model.debug['y_out'], only_inputs=True)
#                 grad(loss, model.debug['rnn'], only_inputs=True)
#                 grad(loss, model.debug['enc_h_mu'], only_inputs=True)
                loss.backward()
                opt.step()
                idx +=1
                batch_loss += loss.item()
                
            cur_mse = batch_loss/float(idx)
            training_loss.append(cur_mse)
            
            if epoch%eval_freq==0:
                self.eval()
                (nelbo, nll, kl), eval_loss = self.forward(*valid_loader.dataset.tensors, anneal = 1.,
                                                          has_missing=val_has_missing,XY=val_newXY, 
                                                       all_seq_lengths=val_all_seq_lengths)
                nelbo, nll, kl = nelbo.item(), nll.item(), kl.item()
                
                if nelbo<best_nelbo:
                    best_nelbo  = nelbo; best_nll = nll; best_kl = kl; best_ep = epoch
                    if fname is not None:
                        torch.save(self.state_dict(), fname)
                        
                if epoch_debug:
                    test_nelbo.append(nelbo)
                    test_nll.append(nll)
                    test_kl.append(kl)
                    
                    
    #                 if kl < 0.:
    #                     print('%.3f' % kl,)

                    train_Y, train_S, train_X, train_M, train_T = train_loader.dataset.tensors
                    test_Y, test_S, test_X, test_M, test_T      = valid_loader.dataset.tensors

                    """
                    step 1: get z using mu not sampling
                    step 2: K-means cluster these z and save centers
                    step 3: return theta_k = g1(z_k) for K clusters
                    """
                    train_z, _   = self.get_mu(train_X,train_Y)
                    train_z      = train_z.detach().numpy()

    #                 likelihood = self.imp_sampling(train_X, train_Y)

    #                 train_likelihood.append(likelihood)

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
                    
#         print ('Best NELBO:%.3f, NLL:%.3f, KL:%.3f@ epoch %d'%(best_nelbo, best_nll, best_kl, best_ep))
        self.best_nelbo    = best_nelbo
        self.best_nll      = best_nll
        self.best_kl       = best_kl
        self.best_ep       = best_ep
        
        if fname is not None and epochs > eval_freq:
            print('loaded state_dict. nelbo: %.4f (ep %d)' % (best_nelbo, best_ep))
            self.load_state_dict(torch.load(fname))
            self.eval()

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
#                        'train_affinity_num_clusters': train_affinity_num_clusters,
#                        'test_affinity_num_clusters': test_affinity_num_clusters,
                       'train_M_sum': train_M.sum(),
                       'test_M_sum': test_M.sum()
                      }
            pickle.dump(results, f)
            f.close()
        
        return best_nelbo, best_nll, best_kl, best_ep
        
class TwoLayer(nn.Module):
    def __init__(self,dim_input, dim_inner, dim_output):
        super(TwoLayer, self).__init__()
        self.fc1 = nn.Linear(dim_input,dim_inner)
        self.fc2 = nn.Linear(dim_inner,dim_output)
        
    def forward(self, x):
        x = self.fc2(F.relu(self.fc1(x)))
        return x
    
class Sublign(Model):
    def __init__(self, dim_stochastic, dim_hidden, dim_rnn, C=0.0, dim_biomarkers=3, 
                 reg_type = 'l2', sigmoid=True, learn_time=True, auto_delta=True, max_delta=10.,
                plot_debug=False, epoch_debug=False, beta=0.001, device='cpu',
                how_missing='linear'):
        """
        note no lr here. lr is in fit.
        """
        super(Sublign, self).__init__()
        self.dim_stochastic = dim_stochastic
        self.dim_hidden     = dim_hidden 
        self.dim_rnn        = dim_rnn
        self.n_biomarkers   = dim_biomarkers
        self.C              = C
        self.reg_type       = reg_type
        
        self.sigmoid        = sigmoid
        
        self.dz_features    = self.dim_stochastic
        rnn_input_size      = self.n_biomarkers + 1
        
        self.subtypes_km = None
        self.rnn       = nn.RNN(rnn_input_size, self.dim_rnn, 1, batch_first = True)
        self.enc_h_mu  = nn.Linear(self.dim_rnn, self.dim_stochastic)
        self.enc_h_sig = nn.Linear(self.dim_rnn, self.dim_stochastic)
        
        self.how_missing = how_missing

        # initialize functions theta = g1(z)
        if self.sigmoid:
            self.dec_z_beta0 = TwoLayer(self.dz_features, self.dim_hidden, self.n_biomarkers)
            self.dec_z_beta1 = TwoLayer(self.dz_features, self.dim_hidden, self.n_biomarkers)
        else:
            self.dec_z_a = TwoLayer(self.dz_features, self.dim_hidden, self.n_biomarkers)
            self.dec_z_b = TwoLayer(self.dz_features, self.dim_hidden, self.n_biomarkers)
            self.dec_z_c = TwoLayer(self.dz_features, self.dim_hidden, self.n_biomarkers)
    
        # experiments for delta
        if auto_delta:
            self.max_delta      = 10.
            self.auto_delta     = True
            self.learn_time     = True
        elif learn_time:
            self.max_delta      = max_delta
            self.auto_delta     = False
            self.learn_time     = True
        else:
            self.max_delta      = 0.
            self.auto_delta     = False
            self.learn_time     = False
        
        if not learn_time:
            self.learn_time = False
            self.max_delta = 0.
            self.auto_delta = False
            
        self.N_delta_bins       = 50
        
        if device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
            
        self.debug  = {}
        self.beta   = beta
        self.data_num = 1
        
    def get_delta_options(self, Xvals):
        # output delta_options is tensor size N_patients, N_delta_bins
        N_patients = Xvals.shape[0]
        
        if self.auto_delta:
            max_time_patient = Xvals.max(axis=1).values
            max_time_all = max_time_patient.max()
            max_delta_patient = max_time_all - max_time_patient
            delta_options = torch.zeros(N_patients,self.N_delta_bins).to(self.device)
            for i in range(N_patients):
                delta_options[i] = torch.linspace(0,max_delta_patient[i,0],self.N_delta_bins)
                
            return delta_options
        else:
            delta_options = torch.linspace(0, self.max_delta, self.N_delta_bins)
            return delta_options[None,:].repeat(N_patients, 1).to(self.device)
    
    def calc_loss_per_delta(self, X, Y, M, theta, delta_options, kl):
        """
        input:
         - X (N_patients, N_visits, 1)
         - Y (N_patients, N_visits, N_biomarkers)
         - theta (N_patients, N_biomarkers each component) 
         - delta_options (N_patients, N_delta_bins)
        
        output:
         - loss_per_patient (N_patients, N_delta_bins)
         
        step 1: convert everything to size N_patients, N_visits, N_biomarkers, N_delta_bins
        step 2: calculate loss yhat = f(x+delta; theta) 
        """
        N_patients, N_visits, N_biomarkers = Y.shape
        
        
        X_repeat          = X[:,:,:,None].repeat(1,1,N_biomarkers,self.N_delta_bins)
        Y_repeat          = Y[:,:,:,None].repeat(1,1,1,self.N_delta_bins)
        delta_opt_repeat  = delta_options[:,None,None,:].repeat(1,N_visits,N_biomarkers,1)
        
        if self.sigmoid:
            beta0 = theta[0][:,None,:,None].repeat(1,N_visits,1,self.N_delta_bins)
            beta1 = theta[1][:,None,:,None].repeat(1,N_visits,1,self.N_delta_bins)
    
            sig_input = X_repeat + delta_opt_repeat
            mm        = torch.nn.Sigmoid()
            mm_input  = (beta0 + beta1 * sig_input).to(self.device)
            yhat      = mm(mm_input)
        else:
            a = theta[0][:,None,:,None].repeat(1,N_visits,1,self.N_delta_bins)
            b = theta[1][:,None,:,None].repeat(1,N_visits,1,self.N_delta_bins)
            c = theta[2][:,None,:,None].repeat(1,N_visits,1,self.N_delta_bins)
            
            quad_input = X_repeat + delta_opt_repeat
            yhat       = quad_function(a,b,c,quad_input)
            
        kl_repeat = kl[:,None].repeat(1,self.N_delta_bins)
        loss      = ((yhat - Y_repeat)**2)
        
        M_repeat = M[:,:,:,None].repeat(1,1,1,self.N_delta_bins)
        loss = loss.masked_fill(M_repeat == 0., 0.)
        
        loss_sum  = loss.sum(axis=1).sum(axis=1)
        
        delta_term = torch.log(torch.ones_like(loss_sum) / self.N_delta_bins).to(self.device)
        
        kl_repeat = kl_repeat.to(self.device)
        return loss_sum + self.beta*kl_repeat + delta_term
    
    def get_best_delta(self, X,Y,M,theta, kl):
        """
        output: best_delta is size N_patients
        
        step 1: if subnolign, return 0.
        step 2: get all the delta options
        step 3: calculate loss for each option
        step 4: find best delta option
        
        note that z could be either from sample or get_mu so not included here
        """
        
        # TODO: interpolate X and Y if they're missing
        
        if type(X) == np.ndarray:
            X = torch.tensor(X).to(self.device)
            Y = torch.tensor(Y).to(self.device)
            M = torch.tensor(M).to(self.device)
            
        N = X.shape[0]
        
        if not self.learn_time:
            return torch.zeros(N)
        
        delta_options  = self.get_delta_options(X)
        
        loss_per_delta = self.calc_loss_per_delta(X,Y,M,theta, delta_options, kl)
        
        min_delta      = loss_per_delta.min(axis=1).indices
        
        best_delta = torch.zeros(N).to(self.device)
        for i in range(N):
            best_delta[i] = delta_options[i][min_delta[i]]
        return best_delta
    
    def predict_Y(self, X,Y,theta,delta):
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
        N_patients, N_visits, N_biomarkers = Y.shape
        
        X_repeat   = X.repeat(1,1,N_biomarkers)
        delta_rep  = delta[:,None,None].repeat(1,N_visits,N_biomarkers)
        
        if self.sigmoid:
            beta0 = theta[0][:,None,:].repeat(1,N_visits,1)
            beta1 = theta[1][:,None,:].repeat(1,N_visits,1)
            
            sig_input = X_repeat + delta_rep
            mm        = torch.nn.Sigmoid()
            mm_input  = (beta0 + beta1 * sig_input).to(self.device)
            yhat      = mm(mm_input)
        else:
            a = theta[0][:,None,:].repeat(1,N_visits,1)
            b = theta[1][:,None,:].repeat(1,N_visits,1)
            c = theta[2][:,None,:].repeat(1,N_visits,1)
            
            quad_input = X_repeat + delta_rep
            yhat       = quad_function(a,b,c,quad_input)
        return yhat
    
    def get_loss(self, Y, S, X, M, anneal=1.,
                XY=None,all_seq_lengths=None, has_missing=False):
        
        if type(X) == np.ndarray:
            X = torch.tensor(X).to(self.device)
            Y = torch.tensor(Y).to(self.device)
            M = torch.tensor(M).to(self.device)
            
        z, kl        = self.sample(X,Y,XY=XY,
                                   all_seq_lengths=all_seq_lengths, has_missing=has_missing)
        theta        = self.infer_functional_params(z)
        with torch.no_grad():
            best_delta   = self.get_best_delta(X,Y,M,theta, kl)
        yhat        = self.predict_Y(X,Y,theta,best_delta)
        self.debug['y_out'] = yhat
        squared     = (Y - yhat)**2
        
        # mask out originally missing values
        squared     = squared.masked_fill(M == 0., 0)
        nll         = squared.sum(-1).sum(-1)
        delta_term  = torch.log(torch.ones_like(nll) / self.N_delta_bins)    
#         nelbo       = nll + self.beta*anneal*kl + delta_term
        nelbo       = nll + self.beta*anneal*kl
        return nelbo, nll, kl
    
    def forward(self, Y, S, X, M, T, anneal = 1., 
                XY=None,all_seq_lengths=None, has_missing=False):
        if type(M) == np.ndarray:
            X = torch.tensor(X).to(self.device)
            Y = torch.tensor(Y).to(self.device)
            M = torch.tensor(M).to(self.device)
            
        if XY is None and (check_has_missing(X) or check_has_missing(Y)):
            has_missing = True
            XY = torch.cat([X,Y], axis=2)
            newXY, all_seq_lengths = convert_XY_pack_pad(XY, how=self.how_missing)
        else:
            has_missing = False
            
        (nelbo, nll, kl) = self.get_loss(Y, S, X, M, anneal = anneal, XY=XY,all_seq_lengths=all_seq_lengths, has_missing=has_missing)
        reg_loss         = nelbo
        for name,param in self.named_parameters():
            reg_loss += self.C*self.apply_reg(param, reg_type=self.reg_type)
        
        normalizer = torch.sum(M)
        norm_nelbo = (torch.sum(nelbo) / normalizer)
        norm_nll   = (torch.sum(nll)/normalizer)
        norm_kl    = torch.mean(kl)
        norm_reg   = torch.sum(reg_loss) / normalizer
        
        return (norm_nelbo, norm_nll, norm_kl), norm_reg
    
    def sample(self, X,Y,mu_std=False,XY=None,all_seq_lengths=None, has_missing=False):
        """
        Returns z and KL sampled from observed X,Y
        """
        cacheXY = XY
        
        if type(X) == np.ndarray:
            X = torch.tensor(X).to(self.device)
            Y = torch.tensor(Y).to(self.device)
            
        XY = torch.cat([X,Y], axis=2)
        
#         import pdb; pdb.set_trace()
        if has_missing:
#             batch_in, sequences = convert_XY_pack_pad(XY,how=self.how_missing)
            pack = torch.nn.utils.rnn.pack_padded_sequence(cacheXY, all_seq_lengths, batch_first=True, enforce_sorted=False)
            _, hidden = self.rnn(pack)
        elif check_has_missing(XY):
            batch_in, sequences = convert_XY_pack_pad(XY,how=self.how_missing)
            pack = torch.nn.utils.rnn.pack_padded_sequence(cacheXY, all_seq_lengths, batch_first=True, enforce_sorted=False)
            _, hidden = self.rnn(pack)
        else:
            _, hidden = self.rnn(XY)
        
        self.debug['rnn'] = hidden
        
        hid = torch.squeeze(hidden)
        hid = hid.to(self.device)
        # idx contains list of indices representing the current datapoints in X
        # mu_param is a pytorch tensor (randomly initialized) of size N x dimensionality of latent space
        # gamma = 1 (learning w/ inf. network) or 0. (learning w/ svi) 
        
        mu_table = mu_param[idx]
        mu_enc     = self.enc_h_mu(hid)
        mu = gamma*mu_enc+(1-gamma)*mu_table

        sig    = torch.exp(self.enc_h_sig(hid))
        q_dist = Independent(Normal(mu, sig), 1)
        z      = torch.squeeze(q_dist.rsample((1,)))
        p_dist = Independent(Normal(torch.zeros_like(mu), torch.ones_like(sig)), 1)
        kl     = q_dist.log_prob(z)-p_dist.log_prob(z)
        
        self.debug['hid'] = hid
        self.debug['kl'] = kl
        self.debug['mu'] = mu
        self.debug['sig'] = sig
        
        if mu_std:
            return z, kl, mu
        else:
            return z, kl
    
    def get_mu(self, X,Y):
        N = X.shape[0]
        if type(X) == np.ndarray:
            X = torch.tensor(X).to(self.device)
            Y = torch.tensor(Y).to(self.device)
        XY = torch.cat([X,Y], axis=2)
        
        if check_has_missing(XY):
            batch_in, sequences = convert_XY_pack_pad(XY)
            pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, sequences, batch_first=True, enforce_sorted=False)
            _, hidden = self.rnn(pack)
        else:
            _, hidden = self.rnn(XY)
            
        hid = torch.squeeze(hidden)
        mu     = self.enc_h_mu(hid)
        return mu, torch.zeros(N)
    
    def infer_functional_params(self, z):
        if self.sigmoid:
            return [self.dec_z_beta0(z), self.dec_z_beta1(z)]
        else:
            return [self.dec_z_a(z), self.dec_z_b(z), self.dec_z_c(z)]

    def get_subtypes(self, X, Y, K=2):
        """
        step 1: get z using mu not sampling
        step 2: K-means cluster these z and save centers
        step 3: return theta_k = g1(z_k) for K clusters
        """
        z, _   = self.get_mu(X,Y)
        if z.get_device() > -1:
            z      = z.cpu().detach().numpy()
        else:
            z      = z.detach().numpy()
        
        # for different cluster algs, plot labels and true subtypes
        km = KMeans(n_clusters=K)
        if np.isnan(z).any():
            print('z has nan in it')
            import pdb; pdb.set_trace()
        km.fit(z)
        self.subtypes_km = km
        
        z_mus  = km.cluster_centers_
        N_dims = Y.shape[2]
        
        if self.sigmoid:
            cent_lst = np.zeros((K,N_dims,2))
        else:
            cent_lst = np.zeros((K,N_dims,3))

        for k_ix in range(K):
            z_mu = z_mus[k_ix]
            z_mu = torch.tensor(z_mu[None,:]).to(self.device)
            theta = self.infer_functional_params(z_mu)
            if theta[0].get_device() > -1:
                theta = [t.cpu().detach().numpy() for t in theta]
            else:
                theta = [t.detach().numpy() for t in theta]

            for param_i, param_component in enumerate(theta):
                for dim_i, dim_val in enumerate(param_component[0]):
                    cent_lst[k_ix,dim_i,param_i] = dim_val
        return cent_lst
    
        
    
    def get_param_subtypes(self, X, Y, K=2):
        """
        step 1: get z using mu not sampling
        step 2: K-means cluster these z and save centers
        step 3: return theta_k = g1(z_k) for K clusters
        """
        params = self.get_params(X,Y)
        pdb
        z      = z.detach().numpy()
        
        # for different cluster algs, plot labels and true subtypes
        km = KMeans(n_clusters=K)
        km.fit(z)
        self.subtypes_km = km
        
        z_mus    = km.cluster_centers_
        cent_lst = list()
        
        for k_ix in range(K):
            z_mu = z_mus[k_ix]
            z_mu = torch.tensor(z_mu[None,:]).to(self.device)
            theta = self.infer_functional_params(z_mu)
            theta = [t.detach().numpy() for t in theta]
            cent_lst.append(theta)
        return cent_lst
    
    def get_params(self, X, Y):
        """
        different from get_subtypes because now there is one theta per person 
        NOT num subtypes
        """
        z, _   = self.get_mu(X,Y)
#         z      = z.detach().numpy()
        if self.sigmoid:
            return [self.dec_z_beta0(z), self.dec_z_beta1(z)]
        else:
            return [self.dec_z_a(z), self.dec_z_b(z), self.dec_z_c(z)]
    
    def get_labels(self, data_dict):
        X = torch.tensor(data_dict['obs_t_collect']).to(self.device)
        Y = torch.tensor(data_dict['Y_collect']).to(self.device)
        
        z, _   = self.get_mu(X,Y)
        
        if z.get_device() > -1:
            z = z.cpu().detach().numpy()
        else:
            z = z.detach().numpy()
        
        labels = self.subtypes_km.predict(z)
        return labels
    
    def get_deltas(self, data_dict):
        X = torch.tensor(data_dict['obs_t_collect']).to(self.device)
        Y = torch.tensor(data_dict['Y_collect']).to(self.device)
        M = torch.tensor(data_dict['mask_collect']).to(self.device)
        
        z, kl        = self.get_mu(X,Y)
        theta        = self.infer_functional_params(z)
        
        if type(X) == np.ndarray:
            X = torch.tensor(X).to(self.device)
            Y = torch.tensor(Y).to(self.device)
            M = torch.tensor(M).to(self.device)
            
        best_delta   = self.get_best_delta(X,Y,M,theta, kl)
        return best_delta
        
    def get_mse(self,X,Y,M,theta,best_delta):
        yhat    = self.predict_Y(X,Y,theta,best_delta)
        squared = (Y - yhat)**2
        nll     = squared.sum(-1).sum(-1)
        normsum = torch.sum(M)
        return torch.sum(nll) / normsum
        
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
        
        cent_lst = self.get_subtypes(train_data_dict['obs_t_collect'], train_data_dict['Y_collect'], K=K)
        
        test_X = torch.tensor(test_data_dict['obs_t_collect']).to(self.device)
        test_Y = torch.tensor(test_data_dict['Y_collect']).to(self.device)
        test_M = torch.tensor(test_data_dict['mask_collect']).to(self.device)
        
        test_z, kl   = self.get_mu(test_X,test_Y)
        test_theta   = self.infer_functional_params(test_z)
        best_delta   = self.get_best_delta(test_X,test_Y, test_M, test_theta, kl)
        
        test_z        = test_z.detach().numpy()
        test_clusters = self.subtypes_km.predict(test_z)
        true_clusters = [int(i) for i in np.squeeze(test_data_dict['s_collect'])]
        
        test_M     = torch.ones_like(test_X)
        test_mse   = self.get_mse(test_X, test_Y, test_M, test_theta, best_delta)
        test_ari   = adjusted_rand_score(test_clusters, true_clusters)

        test_swaps = get_cluster_swap_metric(test_clusters, test_data_dict['t_collect'][:,0,0], best_delta.detach().numpy())
        test_pear  = get_cluster_pear_metric(test_clusters, test_data_dict['t_collect'][:,0,0], best_delta.detach().numpy())
        
        results = {
            'mse': test_mse,
            'ari': test_ari,
            'swaps': test_swaps,
            'pear': test_pear,
            'cent_lst': cent_lst
        }
        
        return results

    def imp_sampling(self, X, Y, imp_samples=10, delta_gran = 20):
        delta_gran = self.N_delta_bins
        if type(X) == np.ndarray:
            X = torch.tensor(X).to(self.device)
            Y = torch.tensor(Y).to(self.device)
            
        ll_estimates   = torch.zeros((imp_samples,delta_gran,X.shape[0])).to(X.device)
        ll_priors      = torch.zeros((imp_samples,delta_gran,X.shape[0])).to(X.device)
        ll_posteriors  = torch.zeros((imp_samples,delta_gran,X.shape[0])).to(X.device)
        
        # TODO: fix this 
        N_latent_dim = self.dz_features
        mu_prior, std_prior = torch.zeros(N_latent_dim), torch.ones(N_latent_dim)
        
        M = torch.ones_like(Y)
        for sample in range(imp_samples):
            z, kl, qz_mu = self.sample(X,Y,mu_std=True)
            qz_sig       = torch.ones(N_latent_dim)
            
            theta        = self.infer_functional_params(z)
            ll_estimate_list, ll_posterior_list, ll_prior_list = [],[],[]
            for dval in np.linspace(0,5,delta_gran):
                best_delta = self.get_best_delta(X,Y,M,theta, kl)
                dval       = best_delta*0.+dval
                #print (best_delta.shape, dval)
                #best_delta   = dval
                yhat        = self.predict_Y(X,Y,theta,best_delta)
                nll = (yhat - Y) ** 2
                ll_estimate_list.append(-1*nll.sum(-1).sum(-1))
                ll_prior_list.append((-1*self.masked_gaussian_nll_3d(z, mu_prior, std_prior)).sum(-1))
                ll_posterior_list.append((-1*self.masked_gaussian_nll_3d(z, qz_mu, qz_sig)).sum(-1)) 
            ll_priors[sample]     = torch.stack(ll_prior_list)
            ll_estimates[sample]  = torch.stack(ll_estimate_list)
            ll_posteriors[sample] = torch.stack(ll_posterior_list)
            
        nll_estimate = -1*(torch.logsumexp(ll_estimates.view(imp_samples*delta_gran,-1) + ll_priors.view(imp_samples*delta_gran,-1) - ll_posteriors.view(imp_samples*delta_gran,-1), dim=0) - np.log(imp_samples*delta_gran))
        log_p =  torch.mean(nll_estimate)
        return log_p
    def get_subtypes_datadict(self, data_dict, K=2):
        """
        assumes you've already fit a model
        """
        X = torch.tensor(data_dict['obs_t_collect']).to(self.device)
        Y = torch.tensor(data_dict['Y_collect']).to(self.device)
        M = torch.tensor(data_dict['mask_collect']).to(self.device)
        
        z, _   = self.get_mu(X,Y)
        
        if z.get_device() > -1:
            z = z.cpu().detach().numpy().copy()
        else:
            z = z.detach().numpy().copy()
        
        if self.subtypes_km is None:
            # for different cluster algs, plot labels and true subtypes
            km = KMeans(n_clusters=K)
            km.fit(z)
            self.subtypes_km = km
            
        labels = self.subtypes_km.predict(z)
        return labels
    
def get_hyperparameters(data_format_num):
    if data_format_num < 3:
        C, ds, dh, drnn, reg_type, lr = 0., 10, 20, 50, 'l1', 0.01
    
    if data_format_num == 5 or data_format_num == 3:
        C, ds, dh, drnn, reg_type, lr = 0.01, 20, 20, 100, 'l2', 0.01
#     if data_format_num == 4:
#         C, ds, dh, drnn, reg_type, lr =  0.0, 30, 10, 50, 'l1', 0.001
    if data_format_num == 1:
        C, ds, dh, drnn, reg_type, lr = 0.0, 20, 30, 150, 'l1', 0.001
#         C, ds, dh, drnn, reg_type, lr = 0.0, 20, 20, 100, 'l1', 0.001
    if data_format_num == 11:
        C, ds, dh, drnn, reg_type, lr = 0.0, 20, 30, 150, 'l1', 0.001
    elif data_format_num > 2:
        C, ds, dh, drnn, reg_type, lr = 0., 20, 50, 100, 'l1', 0.01
        
    return C, ds, dh, drnn, reg_type, lr

def main():
    import argparse
    import os
    import sys
    sys.path.append('../data')
    sys.path.append('../plot')
    from load import sigmoid, quadratic, chf, parkinsons, load_data_format
    from data_utils import parse_data, change_missing
    from plot_utils import plot_subtypes, plot_latent
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--epochs', action='store', type=int, default=800, help="Number of epochs")
    parser.add_argument('--trials', action='store', type=int, default=1, help="Number of trials")
    parser.add_argument('--model_name', action='store', type=str, default='SubLign', help="Model name for Latex table making")
    
    # datasets
    parser.add_argument('--data_num', action='store', type=int, help="Data Format Number")
    parser.add_argument('--chf', action='store_true', help="Use CHF dataset")
    parser.add_argument('--ppmi', action='store_true', help="Use PPMI dataset")
    
    # delta setup
#     parser.add_argument('--auto_delta', action='store_true', help="Learn delta dynamically for each patient")
    parser.add_argument('--max_delta', action='store', type=float, help="Maximum possible delta")
    parser.add_argument('--no_time', action='store_true', help="Learn time at all")
    
    # debugging
    parser.add_argument('--verbose', action='store_true', help="Plot everything")
    parser.add_argument('--missing', action='store', type=float, default=0., help="What percent of data to make missing")
    parser.add_argument('--plot_debug', action='store_true', help="Make animated gif about alignment / clusterings over epochs")
    parser.add_argument('--epoch_debug', action='store_true', help="Save pickle about epoch differences over training")
    parser.add_argument('--likelihood', action='store_true', help="Print likelihood")
    parser.add_argument('--lr', action='store', type=float, help="Learning rate override")
    parser.add_argument('--eval_freq', action='store', type=int, help="Make this larger than epochs for faster results", default=25)

    # other experiments
    
    args = parser.parse_args()
    
    trial_results = np.zeros((args.trials, 4))
    
    data_format_num = args.data_num
    
    if args.max_delta is None:
        auto_delta = True
    else:
        auto_delta = False
    for trial_num in range(args.trials):
        # datasets
        if data_format_num is not None:
            max_visits      = 4 
            num_output_dims = 3 if data_format_num < 3 else 1
            use_sigmoid     = data_format_num < 3

            if data_format_num > 10:
                use_sigmoid     = True
                num_output_dims = 3
                
            C, d_s, d_h, d_rnn, reg_type, lr = get_hyperparameters(data_format_num)
            
            if args.lr != None:
                print('Learning rate: %.3f' % args.lr)
                lr = args.lr
                
            data = load_data_format(data_format_num, trial_num, cache=True)
            shuffle = False
        elif args.chf:
            print('HERE2')
            data       = chf()
            max_visits = 38
            shuffle    = True
        elif args.ppmi:
            data       = parkinsons()
            max_visits = 17
            shuffle    = True
#             data = data[data['subtype'] == 1]

        train_data_loader, train_data_dict, _, _, test_data_loader, test_data_dict, valid_pid, test_pid, unique_pid = parse_data(data.values, max_visits=max_visits, test_per=0.2, valid_per=0.2, shuffle=shuffle)
#         train_data_loader, train_data_dict, test_data_loader, test_data_dict, p_ids, full_p_ids = parse_data(data.values, max_visits=max_visits, test_per=0.2, shuffle=shuffle)
        
#         pickle.dump((train_data_loader, train_data_dict, test_data_loader, test_data_dict, p_ids, full_p_ids), open('../synthetic_runs/data.pk', 'wb'))
#         import pickle
#         train_data_loader, train_data_dict, test_data_loader, test_data_dict, p_ids, full_p_ids = pickle.load(open('../synthetic_runs/data.pk', 'rb'))
        if args.missing > 0.:
            train_data_loader, train_data_dict = change_missing(train_data_dict, args.missing)
            
        data_loader, collect_dict, unique_pid = parse_data(data.values, max_visits=max_visits)

        """
        best parmas found through hypertuning (cross_validation/hpsearch.py)
        # sigmoid: C (0.01), dim_h (20), ds (10 mid), dim_rnn (50 mid), reg_type (l1), lr (0.1)
        # quad: C (0.1), dim_h (50), ds (10), dim_rnn (100), reg_type (l1), lr (0.1)
        
        ppmi: (0.0, 10, 10, 50, 'l1', 0.1)
        """
        
        # dim_stochastic, dim_hidden, dim_rnn, C, dim_biomarkers=3, reg_type = 'l2', 
        if data_format_num is not None:
            model = Sublign(d_s, d_h, d_rnn, C, num_output_dims, sigmoid=use_sigmoid, reg_type=reg_type, auto_delta=auto_delta, max_delta=args.max_delta, learn_time=(not args.no_time))
            
            model.fit(train_data_loader, test_data_loader, args.epochs, lr, verbose=args.verbose, fname='runs/data%d_trial%d.pt' % (data_format_num, trial_num), eval_freq=args.eval_freq,epoch_debug=args.epoch_debug, plot_debug=args.plot_debug)
            
        elif args.chf:
            args.verbose = False
            model = Sublign(10, 20, 50, 0.1, data.shape[1] - 4, sigmoid=True, reg_type='l1', auto_delta=True, max_delta=args.max_delta, learn_time=(not args.no_time))
            model.fit(data_loader, data_loader, args.epochs, 0.01, verbose=args.verbose)
            subtypes = model.get_subtypes(collect_dict['obs_t_collect'], collect_dict['Y_collect'], K=3)
            labels = model.get_labels(collect_dict['obs_t_collect'], collect_dict['Y_collect'])
            deltas = model.get_deltas(collect_dict['obs_t_collect'], collect_dict['Y_collect'], collect_dict['mask_collect'])
            zs = model.get_mu(collect_dict['obs_t_collect'], collect_dict['Y_collect'])
            import pickle
            pickle.dump((labels, deltas, subtypes, unique_pid, collect_dict, zs), open('../clinical_runs/chf_sublign_hera3.pk', 'wb')) 
            return
        
        elif args.ppmi:
            args.verbose = False
            # (0.0, 10, 10, 50, 'l1', 0.1)
            # C (0.1), dim_h (50), ds (10), dim_rnn (100), reg_type (l1), lr (0.1)
            model = Sublign(10, 10, 20, 0., data.shape[1] - 4, sigmoid=True, reg_type='l1', auto_delta=True, max_delta=args.max_delta, learn_time=(not args.no_time))
#             model.fit(train_data_loader, test_data_loader, args.epochs, 0.1, verbose=args.verbose)
#             subtypes = model.get_subtypes(train_data_dict['obs_t_collect'], train_data_dict['Y_collect'], K=2)
#             labels = model.get_labels(train_data_dict)
#             deltas = model.get_deltas(train_data_dict)
            
            model.fit(data_loader, data_loader, args.epochs, 0.1, verbose=args.verbose)
            subtypes = model.get_subtypes(collect_dict['obs_t_collect'], collect_dict['Y_collect'], K=3)
            labels = model.get_labels(collect_dict)
            deltas = model.get_deltas(collect_dict)

            
            
#             gt_labels = [int(i) for i in test_data_dict['s_collect'].squeeze()]
#             print('ARI: %.3f' % adjusted_rand_score(gt_labels, labels))
            import pickle
            pickle.dump((labels, deltas, subtypes, unique_pid, collect_dict), open('../clinical_runs/ppmi_sublign_PDonly.pk', 'wb'))
            return
            
        subtypes = model.get_subtypes(train_data_dict['obs_t_collect'], train_data_dict['Y_collect'], K=2)
        train_results = model.score(train_data_dict, train_data_dict)
        test_results = model.score(train_data_dict, test_data_dict)
        
        Y = test_data_dict['Y_collect']
        X = test_data_dict['obs_t_collect']
        M = test_data_dict['mask_collect']
        S = None
        T = None
        
        if args.likelihood:
            log_p = model.imp_sampling(X,Y,imp_samples=50)
            print('Test Liklihood: %.3f' % log_p)
        
        (nelbo, nll, kl), _ = model.forward(Y, S, X, M, T, anneal=1.)
#         def forward(self, Y, S, X, M, T, anneal = 1.):
                
        nelbo, nll, kl = nelbo.mean().detach().numpy(), nll.mean().detach().numpy(), kl.mean().detach().numpy()

        if args.verbose:
            plot_subtypes(subtypes, args.sigmoid, train_data_dict)
            plot_latent(model, test_data_dict)
        trial_results[trial_num] = [test_results['mse'],test_results['ari'], test_results['swaps'], test_results['pear']]
        
    if args.trials == 1:
        
        print('Train: %.3f, %.3f, %.3f, %.3f' % (train_results['mse'], train_results['ari'], train_results['swaps'], train_results['pear']))
        print('Test : %.3f, %.3f, %.3f, %.3f' % (test_results['mse'], test_results['ari'], test_results['swaps'], test_results['pear']))
        print('NELBO: %.3f, NLL: %.3f, KL: %.3f' % (nelbo, nll, kl))
    else:
        line_str = list()
        for i,j in zip(trial_results.mean(axis=0), trial_results.std(axis=0)): 
            line_str.append('%.3f $\\pm$ %.3f' % (i,j))
        print(' & '.join([args.model_name] + line_str) + '\\\\')

        trials_fname = 'runs/%s.txt' % args.model_name
        if not os.path.exists(trials_fname):
            f = open(trials_fname, 'w')
        else:
            f = open(trials_fname, 'a')

#         f.write(' & '.join([args.model_name] + line_str) + '\\\\' + '\n')
#         f.close()
        
if __name__=='__main__':
    main()