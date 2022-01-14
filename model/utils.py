import numpy as np
import pandas as pd
import torch

def check_has_missing(X):
    return (X < -999.).any()


def interpolate(X, m=None, t=None, how='linear'):
    if how == 'linear':
        
        N_patients, N_visits, N_dims = X.shape
        all_seq_lengths = list()

        X_old = X
        X = X_old.copy()

#         if X.get_device() > -1:
#             X_numpy = X.cpu().detach().numpy().copy()
#         else:
#             X_numpy = X.detach().numpy().copy()
        is_null = X < -999.

        visits_to_pad = is_null.all(axis=2)

#         pad visits 
        for n in range(N_patients):
            for m in range(N_visits):
                if visits_to_pad[n,m]:
                    X[n,m] = np.zeros(N_dims)
            seq_len = np.sum(~visits_to_pad[n])
            all_seq_lengths.append(seq_len)
            
        
        visits_to_pad_tile = np.tile(visits_to_pad[:,:,None], (1,1, N_dims))
        X[visits_to_pad_tile] = 0.

        null_vals_to_fill = np.logical_and(is_null, ~visits_to_pad_tile)
        X[null_vals_to_fill] = np.nan

        # If an entire dim row is empty for one patient, fill with means over all values for that dim over all patients
        dim_means = np.nanmean(np.nanmean(X, axis=0),axis=0)

        for i in range(N_patients):
            df = pd.DataFrame(X[i])
            df = df.interpolate(method='linear', axis=0, liREDACTED_direction='both')

            # for which dims and which visits do we need to fill in with means (dim_means)?
            visits_to_fill_idx = np.where(~visits_to_pad[i] == True)[0]
            df_vals = df.values
            dims_to_fill_idx = np.where((df_vals[visits_to_fill_idx,:] == 0).all(axis=0) == True)[0]
            if len(dims_to_fill_idx) > 0:
                for i in visits_to_fill_idx:
                    for j in dims_to_fill_idx:
                        df_vals[i,j] = dim_means[j]

            new_X_i = df_vals.astype('float32')
    #         new_XY_i[np.isnan(new_XY_i)] = 0.
            X[i] = np.array(new_X_i.astype('float32'))
        return X
    
    elif how == 'mice':
        from fancyimpute import IterativeImputer
        N_patients, N_visits, N_dims = X.shape
        newX = np.reshape(X,(N_patients, N_visits*N_dims))
        mice_impute = IterativeImputer()
        newX = mice_impute.fit_transform(newX)
        newX = np.reshape(newX,(N_patients, N_visits,N_dims))
        return newX
                          
    elif how == 'mrnn':
        import sys
        sys.path.append('../MRNN')
        from mrnn import mrnn
        
        
        # remove tmp folder
        import os
        import shutil
        dirpath = 'tmp/'
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
            
        x = X
        n_patients, n_visits, n_dims = x.shape
        t2 = np.tile(t,(1,1,n_dims)).squeeze()
        
        N_patients, N_visits, N_dims = x.shape
        
        model_parameters = {'h_dim': 10,
                  'batch_size': 128,
                  'iteration': 1000, 
                  'learning_rate': 0.1}
        
        
        if m is None:
            print('ERROR: m is None. Must input to interpolate function.')
        if torch.is_tensor(x):
            x  = x.numpy()
        if torch.is_tensor(m):
            m  = m.numpy()
        if torch.is_tensor(t2):
            t2 = t2.numpy()
        # Fit mrnn_model
        mrnn_model = mrnn(x, model_parameters)
        mrnn_model.fit(x, m, t2)
        # Impute missing data
        imputed_x = mrnn_model.transform(x, m, t2)
        return imputed_x
    
def convert_XY_pack_pad(XY, how='linear'):
    """
    input:
     - XY (Tensor, size N_patients x N_visits x N_dims + 1): concatenated array of X and Y
     
    output:
     - XY (tensor): same as input XY with all -1000 entries linearly interpolated (or other missingness method)
     - all_seq_lengths (list of len N_patients): length of non-zero visits for each patient
     
    assumes:
     - XY NaN values must be at the end of the visits. [1, NaN, NaN] and NEVER [1, NaN, 1].
       otherwise rnn will break.
       
    Steps:
     1) Compute sequence lengths of observed values
     2) TODO: Linearly interpolate missing values
     3) Return XY, seq lengths
    """
    
    N_patients, N_visits, N_dims = XY.shape
    all_seq_lengths = list()

    XY_old = XY.clone()
    
    if XY.get_device() > -1:
        XY_numpy = XY.cpu().detach().numpy().copy()
    else:
        XY_numpy = XY.detach().numpy().copy()
    is_null = XY_numpy < -999.
    
    visits_to_pad = is_null.all(axis=2)
    
    # pad visits 
    for n in range(N_patients):
        for m in range(N_visits):
            if visits_to_pad[n,m]:
                XY[n,m] = torch.zeros(N_dims)
        seq_len = np.sum(~visits_to_pad[n])
        all_seq_lengths.append(seq_len)
    
    if how == 'linear':
        # linearly interpolate missing values
        visits_to_pad_tile = np.tile(visits_to_pad[:,:,None], (1,1, N_dims))
        XY_numpy[visits_to_pad_tile] = 0.

        null_vals_to_fill = np.logical_and(is_null, ~visits_to_pad_tile)
        XY_numpy[null_vals_to_fill] = np.nan

        # If an entire dim row is empty for one patient, fill with means over all values for that dim over all patients
        dim_means = np.nanmean(np.nanmean(XY_numpy, axis=0),axis=0)

        for i in range(N_patients):
            df = pd.DataFrame(XY_numpy[i])
            df = df.interpolate(method='linear', axis=0, liREDACTED_direction='both')
            
            # for which dims and which visits do we need to fill in with means (dim_means)?
            visits_to_fill_idx = np.where(~visits_to_pad[i] == True)[0]
            df_vals = df.values
            dims_to_fill_idx = np.where((df_vals[visits_to_fill_idx,:] == 0).all(axis=0) == True)[0]
            if len(dims_to_fill_idx) > 0:
                for i in visits_to_fill_idx:
                    for j in dims_to_fill_idx:
                        df_vals[i,j] = dim_means[j]

            new_XY_i = df_vals.astype('float32')
    #         new_XY_i[np.isnan(new_XY_i)] = 0.
            XY[i] = torch.Tensor(new_XY_i.astype('float32'))
        return XY, all_seq_lengths
    elif how == 'mice':
        X = XY[:,:,:1]
        Y = XY[:,:,1:]

        newY = interpolate(Y, how='mice')
        newY = torch.Tensor(newY.astype('float32'))

        newXY = torch.cat([X,newY], axis=2)
        return newXY, all_seq_lengths
    
    elif how == 'mrnn':            
        # make x, t, m
        t = XY[:,:,:1]
        x = XY[:,:,1:]
        x[x == -1000]  = np.nan
        
        m = 1 - np.isnan(x)
        x[np.isnan(x)] = 0
        
        imputed_x = interpolate(x, m, t, how='mrnn')
        
        t = torch.tensor(t)
        imputed_x = torch.tensor(imputed_x)
        newXY = torch.cat([t,imputed_x], axis=2)
        return newXY, all_seq_lengths
    

def quad_function(a,b,c,X):
        return a*X*X + b*X + c