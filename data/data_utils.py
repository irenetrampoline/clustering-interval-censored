import collections
import numpy as np
import numpy.ma as ma
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split


def check_has_missing(X):
    return (X == -1000).any()

def replace_with_1000(X):
    X[np.isnan(X)] = -1000
    
def replace_with_mean(Y):
    REDACTED = np.where(np.isnan(Y), ma.array(Y, mask=np.isnan(Y)).mean(axis=0), Y)    
    return REDACTED

def parse_data(raw_data, max_visits=3, test_per=None, shuffle=False, valid_per=None, device='cpu'):
    assert(raw_data.shape[1] >= 5)
    """
    Takes pandas.DataFrame and:
    1) turns into tensors
    2) fills empty/nan entries with -1000
    
    M: 1 if value IS observed, 0 if not.
    """
    N_dims = raw_data.shape[1] - 4
    Y       = raw_data[:, :N_dims]
    subtype = raw_data[:,-4]
    time    = raw_data[:,-3]
    pids     = raw_data[:,-2]
    obs_t   = raw_data[:,-1]

    Y_collect = []
    s_collect = []
    t_collect = []
    obs_t_collect = []
    mask_collect = []

    unique_pid = np.unique(pids)
    N_patients = len(unique_pid)
    
    
    cnter = collections.Counter(pids)
    max_visits = cnter.most_common(1)[0][1]
    print('Max visits: %d' % max_visits)
    
    Y_collect = np.zeros((N_patients, max_visits, N_dims)).astype('float32')
    t_collect = np.zeros((N_patients, max_visits,1)).astype('float32')
    s_collect = np.zeros((N_patients,1)).astype('float32')
    obs_t_collect = np.zeros((N_patients, max_visits,1)).astype('float32')
    mask_collect = np.zeros((N_patients, max_visits, N_dims)).astype('float32')
    
    
    Y_collect.fill(np.nan)
    t_collect.fill(np.nan)
    s_collect.fill(np.nan)
    obs_t_collect.fill(np.nan)
    
    for pix, p in enumerate(unique_pid):
        
        # find pt data
        idx   = np.argwhere(p==pids).reshape(-1)
        # collect subtype
    #     assert np.unique(subtype[idx]).shape[0]==1,'should only be a single subtype'

        # collect patient times
        pt_time = time[idx]
        obs_time = obs_t[idx]

        timesort_idx= np.argsort(pt_time)
        
        # collect patient data
        pt_data = Y[idx]
        
        num_visits = len(idx)
        
        for v in range(num_visits):
            for d in range(N_dims):
                val = pt_data[timesort_idx][v,d]
                Y_collect[pix, v, d] = val
                if np.isnan(val): 
                    mask_collect[pix,v,d] = 0
                else:
                    mask_collect[pix,v,d] = 1
            t_collect[pix,v] = pt_time[timesort_idx][v]
            obs_t_collect[pix,v] = obs_time[timesort_idx][v]
        s_collect[pix] = subtype[idx][0]
    
    # TODO: replace with means
    
    replace_with_1000(Y_collect)
    replace_with_1000(t_collect)
    replace_with_1000(s_collect)
    replace_with_1000(obs_t_collect)

#     Y_collect = replace_with_mean(Y_collect)
#     t_collect = replace_with_mean(t_collect)
#     s_collect = replace_with_mean(s_collect)
#     obs_t_collect = replace_with_mean(obs_t_collect)
    
    Y_collect = Y_collect.astype('float32')
    t_collect = t_collect.astype('float32')
    s_collect = s_collect.astype('float32')
    obs_t_collect = obs_t_collect.astype('float32')
    
    if test_per and valid_per is None:
        train_idx, test_idx = train_test_split(range(N_patients), test_size=test_per, shuffle=shuffle)
        train_dict = {
            'Y_collect': Y_collect[train_idx],
            't_collect': t_collect[train_idx],
            's_collect': s_collect[train_idx],
            'obs_t_collect': obs_t_collect[train_idx],
            'mask_collect': mask_collect[train_idx]
        }
        train_data_loader = get_loader(Y_collect[train_idx], s_collect[train_idx], obs_t_collect[train_idx], mask_collect[train_idx], t_collect[train_idx], device=device)
        
        test_dict = {
            'Y_collect': Y_collect[test_idx],
            't_collect': t_collect[test_idx],
            's_collect': s_collect[test_idx],
            'obs_t_collect': obs_t_collect[test_idx],
            'mask_collect': mask_collect[test_idx]
        }
        test_data_loader = get_loader(Y_collect[test_idx], s_collect[test_idx], obs_t_collect[test_idx], mask_collect[test_idx], t_collect[test_idx], device=device)
        
        return train_data_loader, train_dict, test_data_loader, test_dict, unique_pid[test_idx], unique_pid
    elif test_per and valid_per:
        train_per = 1. - test_per - valid_per
        valid_per_normalized = valid_per / (test_per + valid_per)
        train_idx, valid_test_idx = train_test_split(range(N_patients), train_size=train_per, shuffle=shuffle)
        valid_idx, test_idx = train_test_split(range(len(valid_test_idx)), train_size=valid_per_normalized, shuffle=shuffle)
        
        train_dict = {
            'Y_collect': Y_collect[train_idx],
            't_collect': t_collect[train_idx],
            's_collect': s_collect[train_idx],
            'obs_t_collect': obs_t_collect[train_idx],
            'mask_collect': mask_collect[train_idx]
        }
        train_data_loader = get_loader(Y_collect[train_idx], s_collect[train_idx], obs_t_collect[train_idx], mask_collect[train_idx], t_collect[train_idx], device=device)
        
        valid_dict = {
            'Y_collect': Y_collect[valid_idx],
            't_collect': t_collect[valid_idx],
            's_collect': s_collect[valid_idx],
            'obs_t_collect': obs_t_collect[valid_idx],
            'mask_collect': mask_collect[valid_idx]
        }
        valid_data_loader = get_loader(Y_collect[valid_idx], s_collect[valid_idx], obs_t_collect[valid_idx], mask_collect[valid_idx], t_collect[valid_idx], device=device)
        
        test_dict = {
            'Y_collect': Y_collect[test_idx],
            't_collect': t_collect[test_idx],
            's_collect': s_collect[test_idx],
            'obs_t_collect': obs_t_collect[test_idx],
            'mask_collect': mask_collect[test_idx]
        }
        test_data_loader = get_loader(Y_collect[test_idx], s_collect[test_idx], obs_t_collect[test_idx], mask_collect[test_idx], t_collect[test_idx], device=device)
        
        return train_data_loader, train_dict, valid_data_loader, valid_dict, test_data_loader, test_dict, unique_pid[valid_idx], unique_pid[test_idx], unique_pid
    else:
        collect_dict = {
            'Y_collect': Y_collect,
            't_collect': t_collect,
            's_collect': s_collect,
            'obs_t_collect': obs_t_collect,
            'mask_collect': mask_collect
        }

        data_loader = get_loader(Y_collect, s_collect, obs_t_collect, mask_collect, t_collect, device=device)
        return data_loader, collect_dict, unique_pid
    
def get_loader(Y_collect, s_collect, obs_t_collect, mask_collect, t_collect, device='cpu'):
    """
    loader has five components: Y, s, X, M, T
    """
    if device == 'cpu':
        device_torch = torch.device('cpu')
    else:
        device_torch = torch.device('cuda')

    train_data        = TensorDataset(torch.from_numpy(Y_collect).to(device_torch), 
                                    torch.from_numpy(s_collect).to(device_torch), 
                                    torch.from_numpy(obs_t_collect).to(device_torch),
                                      torch.from_numpy(mask_collect).to(device_torch),
                                      torch.from_numpy(t_collect).to(device_torch)
                                     )
    
    data_loader = DataLoader(train_data, batch_size=len(s_collect), shuffle=False)
    return data_loader

# UPDATE: SEE UTILS.PY FOR FUNCTION
# def convert_XY_pack_pad(XY, how='linear'):
#     """
#     input:
#      - XY (Tensor, size N_patients x N_visits x N_dims + 1): concatenated array of X and Y
     
#     output:
#      - XY (tensor): same as input XY with all -1000 entries converted to 0s
#      - all_seq_lengths (list of len N_patients): length of non-zero visits for each patient
     
#     assumes:
#      - XY NaN values must be at the end of the visits. [1, NaN, NaN] and NEVER [1, NaN, 1].
#        otherwise rnn will break.
       
#     Steps:
#      1) Compute sequence lengths of observed values
#      2) TODO: Linearly interpolate missing values
#      3) Return XY, seq lengths
#     """
    
#     N_patients, N_visits, N_dims = XY.shape
#     all_seq_lengths = list()

#     XY_old = XY.clone()
    
#     XY_numpy = XY.detach().numpy().copy()
#     is_null = XY_numpy == -1000
    
#     visits_to_pad = is_null.all(axis=2)
    
#     # pad visits 
#     for n in range(N_patients):
#         for m in range(N_visits):
#             if visits_to_pad[n,m]:
#                 XY[n,m] = torch.zeros(N_dims)
#         seq_len = np.sum(~visits_to_pad[n])
#         all_seq_lengths.append(seq_len)
    
#     # linearly interpolate missing values
#     visits_to_pad_tile = np.tile(visits_to_pad[:,:,None], (1,1, N_dims))
#     XY_numpy[visits_to_pad_tile] = 0.
    
#     null_vals_to_fill = np.logical_and(is_null, ~visits_to_pad_tile)
#     XY_numpy[null_vals_to_fill] = np.nan
    
#     # If an entire dim row is empty for one patient, fill with means over all values for that dim over all patients
#     dim_means = np.nanmean(np.nanmean(XY_numpy, axis=0),axis=0)
    
#     for i in range(N_patients):
#         df = pd.DataFrame(XY_numpy[i])
#         df = df.interpolate(method='linear', axis=0, liREDACTED_direction='both')
        
#         # for which dims and which visits do we need to fill in with means (dim_means)?
#         visits_to_fill_idx = np.where(~visits_to_pad[i] == True)[0]
#         df_vals = df.values
#         dims_to_fill_idx = np.where((df_vals[visits_to_fill_idx,:] == 0).all(axis=0) == True)[0]
#         if len(dims_to_fill_idx) > 0:
#             for i in visits_to_fill_idx:
#                 for j in dims_to_fill_idx:
#                     df_vals[i,j] = dim_means[j]
                    
#         new_XY_i = df_vals.astype('float32')
# #         new_XY_i[np.isnan(new_XY_i)] = 0.
#         XY[i] = torch.Tensor(new_XY_i.astype('float32'))

#     return XY, all_seq_lengths

def change_missing(orig_dict, missing, val=-1000):
    """
    input:
     - orig_dict (dict): contains keys Y_collect, obs_t_collect, t_collect, s_collect
     - missing (float btw 0 and 1): percentage of data to make missing, e.g. 0.1
     
    output:
     - data_loader (torch DataLoader): data loader of Y,s,X
     - data_dict (dict): similar to orig_dict with some visits of patients entered as NaN
    """
    data_dict = orig_dict.copy()
    if missing != 0.:
        N, M, D = data_dict['Y_collect'].shape
        
        Y_collect     = data_dict['Y_collect']
        obs_t_collect = data_dict['obs_t_collect']
        t_collect     = data_dict['t_collect']
        s_collect     = data_dict['s_collect']
        mask_collect  = data_dict['mask_collect']
        
        no_visits_idx = list()
        
        for n in range(N):
            visits_present = list()
            
            for m in range(M):
                # Is this visit missing?
                if random.random() < missing:
                    visits_present.append(m)
            
            for array_idx, v_idx in enumerate(visits_present):
                for d in range(D):
                    data_dict['Y_collect'][n,M-array_idx-1,d] = val
                    
            if len(visits_present) == M:
                no_visits_idx.append(n)
                
        # Remove patients with no visits
        if len(no_visits_idx) > 0:
            
            data_dict['Y_collect']     = np.delete(data_dict['Y_collect'], no_visits_idx, axis=0)
            data_dict['obs_t_collect'] = np.delete(data_dict['obs_t_collect'], no_visits_idx, axis=0)
            data_dict['t_collect']     = np.delete(data_dict['t_collect'], no_visits_idx, axis=0)
            data_dict['s_collect']     = np.delete(data_dict['s_collect'], no_visits_idx, axis=0)
            data_dict['mask_collect']  = np.delete(data_dict['mask_collect'], no_visits_idx, axis=0)
            print('deleted axis', no_visits_idx)
            
        data_loader = get_loader(data_dict['Y_collect'], 
                                 data_dict['s_collect'], 
                                 data_dict['obs_t_collect'],
                                 data_dict['mask_collect']
                                )
        
        return data_loader, data_dict
    else:
        data_loader = get_loader(data_dict['Y_collect'], 
                                 data_dict['s_collect'], 
                                 data_dict['obs_t_collect'],
                                 data_dict['mask_collect']
                                )
        
        return data_loader, data_dict