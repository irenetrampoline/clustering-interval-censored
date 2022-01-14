import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

import torch

def clean_plot():
    ax = plt.subplot(111) 
    ax.spines["top"].set_visible(False) 
    ax.spines["bottom"].set_visible(False) 
    ax.spines["right"].set_visible(False) 
    ax.spines["left"].set_visible(False) 

def plot_delta_comp(true_delta, pred_delta, fname, labels=[], title=None):
    clean_plot()
    
    if len(labels) > 0:
        uniq_labels = np.unique(labels)
        for lab in uniq_labels:
            lab_idx = np.where(labels == lab)[0]
            plt.plot(true_delta[lab_idx], pred_delta[lab_idx],'.')
    else:
        plt.plot(true_delta, pred_delta, '.')
        
    if title:
        plt.title(title)
    
    plt.xlabel('true delta')
    plt.ylabel('predicted delta')
    
    plt.savefig(fname)
    plt.close()
    
def plot_latent_labels(test_z, test_labels, fname, title=None):
    if type(test_z) != np.ndarray:
        test_z      = test_z.detach().numpy()
    
    plt.figure()
    clean_plot()
    N_patients, N_dims = test_z.shape
    N_clusters = len(np.unique(test_labels))
    
    if N_dims == 2:
        for c in range(N_clusters):
            c_ix = np.where(labels == c)[0]
            plt.plot(test_z[c_ix,0], test_z[c_ix,1],'.')
        plt.xlabel('latent dim 1')
        plt.ylabel('latent dim 2')
    else:
        z_transformed = TSNE(n_components=2).fit_transform(test_z)
        for c in range(N_clusters):
            c_ix = np.where(test_labels == c)[0]
            plt.plot(z_transformed[c_ix,0], z_transformed[c_ix,1],'.')
        plt.xlabel('latent dim 1')
        plt.ylabel('latent dim 2')
    
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    
    if title:
        plt.title(title)
    
    plt.savefig(fname)
    plt.close()
    
#     print('Figure saved to %s' % fname)
    
def plot_latent(model, test_data_dict, fname='../figs/latent_test.pdf'):
    device = torch.device('cpu')
    test_X = torch.tensor(test_data_dict['obs_t_collect']).to(device)
    test_Y = torch.tensor(test_data_dict['Y_collect']).to(device)
        
    test_z, _   = model.get_mu(test_X,test_Y)
    test_z      = test_z.detach().numpy()
    test_labels = model.subtypes_km.predict(test_z)
    
    plt.figure()
    clean_plot()
    N_patients, N_dims = test_z.shape
    N_clusters = len(np.unique(test_labels))
    
    if N_dims == 2:
        for c in range(N_clusters):
            c_ix = np.where(labels == c)[0]
            plt.plot(test_z[c_ix,0], test_z[c_ix,1],'.')
        plt.xlabel('z1')
        plt.ylabel('z2')
        plt.savefig(fname)
    else:
        z_transformed = TSNE(n_components=2).fit_transform(test_z)
        for c in range(N_clusters):
            c_ix = np.where(test_labels == c)[0]
            plt.plot(z_transformed[c_ix,0], z_transformed[c_ix,1],'.')
        plt.xlabel('z1')
        plt.ylabel('z2')
        plt.savefig(fname)
    plt.close()
    
    print('Figure saved to %s' % fname)
    
    
    return 

def plot_subtypes(subtypes, is_sigmoid, plot_true=True, fname=None):
    if is_sigmoid:
        plot_sigmoid(subtypes, plot_true, fname=fname)
    else:
        plot_quadratic(subtypes, plot_true, fname=fname)
        
def plot_quadratic(subtypes, plot_true, max_time=4, fname=None):
    """
    Given learned subtypes for sigmoid function, plot them
    """
    K = len(subtypes)
    D = len(subtypes[0][0][0])
    feat_names = [str(i) for i in range(D)]
    # plt.figure(figsize=(12,10))
    ax = plt.subplot(111) 
    colors = ['#ff7f0e','#1f77b4',  '#2ca02c', '#d62728', '#9467bd']
    f_ix = 0
    for c in range(K):
        plot_col_quadratic(ax, subtypes[c][0][0][f_ix], subtypes[c][1][0][f_ix], subtypes[c][2][0][f_ix], max_time, colors[c])
        if plot_true:
            plot_col_quadratic(ax, 2., -7.8, 7.2, max_time, colors[c], ':')
            plot_col_quadratic(ax, 0., 0., 2., max_time, colors[c], ':')
    ax.set_xlim([0,max_time])
    ax.spines["top"].set_visible(False) 
    ax.spines["bottom"].set_visible(False) 
    ax.spines["right"].set_visible(False) 
    ax.spines["left"].set_visible(False) 

    ax.get_xaxis().tick_bottom() 
    ax.get_yaxis().tick_left()
    ax.grid()
    
    if fname==None:
        fname = '../figs/quadratic_subtypes.pdf'
    
    plt.savefig(fname)
    
    print('Figure saved to %s' % fname)
    
    
def plot_sigmoid(subtypes, plot_true, fname=None):
    """
    Given learned subtypes for sigmoid function, plot them
    """
    K = len(subtypes)
    D = len(subtypes[0][0][0])
    max_time = 10
    feat_names = [str(i) for i in range(D)]
    # plt.figure(figsize=(12,10))
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    colors = ['#ff7f0e','#1f77b4',  '#2ca02c', '#d62728', '#9467bd']
    # Plot mean (with shaded std) for each dimension with each subtype (healthy/parkinson's) plotted on the same graph
    for f_ix, (col,ax) in enumerate(zip(feat_names,axs.flatten())):
        for c in range(K):
            plot_col_sigmoid(ax, subtypes[c][0][0][f_ix], subtypes[c][1][0][f_ix],max_time, colors[c])
        ax.title.set_text(col)
        ax.set_xlim([0,max_time])
        ax.spines["top"].set_visible(False) 
        ax.spines["bottom"].set_visible(False) 
        ax.spines["right"].set_visible(False) 
        ax.spines["left"].set_visible(False) 

        ax.get_xaxis().tick_bottom() 
        ax.get_yaxis().tick_left()
        ax.grid()
        
    if fname==None:
        fname = '../figs/sigmoid_subtypes.pdf'
    
    plt.savefig(fname)
    
    print('Figure saved to %s' % fname)
    
# def plot_col(c_ix,data_dict, s_value=None, color='b'):
#     if s_value == s_value:
#         s_idx = np.where(data_dict['s_collect'] == s_value)[0]
#         times = data_dict['t_collect'][s_idx].flatten()
#         vals = data_dict['Y_collect'][s_idx,:,c_ix].flatten()
#         valid_idx = np.where(vals != -1000.)[0]

#     else:
#         s_idx = np.where(data_dict['s_collect'] == s_value)[0]
#         times = data_dict['t_collect'].flatten()
#         vals = data_dict['Y_collect'][:,:,c_ix].flatten()

#         valid_idx = np.where(vals != -1000.)[0]
        
#     val_mean, times1, _ = binned_statistic(times[valid_idx], vals[valid_idx], statistic='mean', bins=20)
#     val_std, times2, _ = binned_statistic(times[valid_idx], vals[valid_idx], statistic='std', bins=20)
    
#     valid_idx = np.where(~np.isnan(val_mean))[0]

#     ax.plot(times1[valid_idx], val_mean[valid_idx], color, linestyle='--')
#     p1 = val_mean[valid_idx] - val_std[valid_idx]
#     p2 = val_mean[valid_idx] + val_std[valid_idx]
#     t = times1[valid_idx]
#     ax.fill_between(t, p1, p2, color=color, alpha=0.1)

def plot_col_sigmoid(ax, sig0, sig1, max_time, color='b'):
    xs = np.linspace(0,max_time, 100)
    ys = [sigmoid(sig0 + sig1*x) for x in xs]
    ax.plot(xs,ys, color, linewidth=5)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_col_quadratic(ax, a, b, c, max_time, color='b',linestyle='-'):
    xs = np.linspace(0,max_time, 100)
    ys = [quad_function(a,b,c,x) for x in xs]
    ax.plot(xs,ys, color, linewidth=5, linestyle=linestyle)
    
def quad_function(a,b,c,X):
    return a*X*X + b*X + c