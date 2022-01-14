"""
make_synthetic_data.py

Generate synthetic data for testing. Should be able to generate data with:

 - 1/2/3 subtypes
 - linear model, nonlinear
 - always / sometimes increasing
 - uniform / normal / other sampling over time
 - F (e.g. 5, 10, 50, 500) features
 - N patients with <= M visits each 
 - 
"""
import numpy as np
import pandas as pd
import pickle as pk
import random

from scipy.optimize import curve_fit
from scipy import interpolate

def sigmoid(x):
    return 1./(1+np.exp(-x))

def load_sigmoid_data(subtypes=2, F=3, N=500, M=4, noise=1,ppmi=False, missing_rate = 0.5):
    """
    inputs:
     - subtypes (int): number of clusters
     - F (int): number of dims
     - N (int): number of patients
     - M (int): number of visits per patient
     - noise (float): how much noise
     - PPMI (bool): use missingness rates similar to PPMI
    output:
     - data
     
    Note: hardcoding supports up to 4 dims and 2 subtypes.
    """
    
    betas = [
        [
        [-4, 1],
        [-1,1.],
        [-8,8],
        [-25, 3.5]
        ],
        [
        [-1,1.],
        [-8,8],
        [-25, 3.5],
        [-4, 1]
        ]
    ]
    
    xs = np.linspace(0,10,100)
    
    data = np.empty((N,M,F))
    patient_subtypes = np.zeros((N,M))
    patient_times = np.zeros((N,M))
    patient_ids = np.zeros((N,M))
    patient_obs_times = np.zeros((N,M))
    
    ppmi_missing_rates = missing_rate * np.ones(4)
    
#     print('generating times, new way')
    for p_id in range(N):
        # draw subtype randomly
        p_subtype = np.random.choice(subtypes)
        
        # OLD TIME
        all_times = np.random.uniform(0,10, size=M)
        all_times = sorted(all_times)
        earliest_patient_time = np.min(all_times)

        # NEW time generating function
        
#         first_times = np.random.uniform(0, 3)
#         all_times = np.empty(M)
#         all_times[0] = first_times
#         for i in range(1, M):
#             all_times[i] = np.random.uniform(first_times, 10)
#         earliest_patient_time = np.min(all_times)
        
        for visit_id in range(M):
            
            patient_subtypes[p_id,visit_id] = p_subtype
            patient_ids[p_id,visit_id] = p_id
            
            t = all_times[visit_id]
            
            patient_times[p_id, visit_id] = t
            
            p_obs = np.empty(F)
            
            all_dim_empty = True
            for f_ix in range(F):
                p_obs = sigmoid(betas[p_subtype][f_ix][0] + betas[p_subtype][f_ix][1]*t) + np.random.normal(scale=noise)
                
                if random.random() > ppmi_missing_rates[f_ix]:
                    data[p_id, visit_id,f_ix] = p_obs
                    all_dim_empty = False
                else:
                    data[p_id, visit_id,f_ix] = None
        
            if all_dim_empty:
                patient_obs_times[p_id,visit_id] = None
            else:
                patient_obs_times[p_id,visit_id] = patient_times[p_id,visit_id] - earliest_patient_time
                
    df_data = np.reshape(data, (N*M, F))
    df_subtypes = np.reshape(patient_subtypes, (N*M, 1))
    df_times = np.reshape(patient_times, (N*M, 1))
    df_ids = np.reshape(patient_ids, (N*M, 1))
    df_obs_times = np.reshape(patient_obs_times, (N*M, 1))
    
    df_data = np.clip(df_data, 0., 1.)
    
    df = pd.DataFrame(df_data)
    df['subtype'] = df_subtypes
    df['time'] = df_times
    df['p_id'] = df_ids
    df['obs_time'] = df_obs_times
    
    return df
    
    
def load_synthetic_data(subtypes=3, model='linear', increasing=False, sampling='uniform', 
                        F=1, N=500,M=4, noise=1., extreme=False, use_godmode=False, obs_time=True, 
                        make_x=False, case1=False, case2=False, case3=False, case4=False, case5=False, 
                        case6=False, case7=False, case8=False, case9=False, case10=False, gap = 0, overlap=True):
    """
    inputs
    ===
     - subtypes: 1-5, number of subtypes
     - model: 
        - 'linear': linear model
        - 'sigmoid': sigmoid model
        - 'none': no time component, just mixture model
     - increasing: True/False, assume increasing (positive slope for linear) or models might not be increasing
     - sampling: 'uniform', 'normal', 'other' sampling over time
     - F: integer, number of features
     - N: integer, number of patients
     - M: integer, number of visits
     - extreme: boolean, make very obvious linear models 
    returns data in a dataframe where rows correspond to each patient visit and columns are features and true underlying subtypes
    
    Under the hood, without loss of generality:
     - assume time goes from T=0 to 4
     - assume all features range from 0 to 10
     
    special cases:
     - makex (bool): makes an X [aka case0]
     - case1 (bool): lines are closer together, have different slopes. Line on top has flatter slope than line on bottom.
     - case2 (bool): lines start together and then diverge
     - case3 (bool): one line is flat and the other starts above, drops below, and then comes back above
     - case 4: just u (cluster)
     - case 6 (ICML submission): one is quadratic and the other is flat, can be overlapping if gap < 0.
     - case 7 (NEW): one is quadratic and the other is linear, can be overlapping if overlap is True
     - case 8 (NEW): both are quadratic in opposite directions, can be overlapping if gap < 0.
     
    outputs
    ====
     - data (dataframe)
     - truelines (list of lists)
    """
    
    
    
    # generate underlying subtypes
    # for each dim, pick a number between 0 and 10
    # pick another number between 0 and 10
    
    godmode = [(5,9.5), (0, 3), (0.5, 4)]
    make_x_coord = [(2, 9.5), (8, 1)]
    case1_coord = [(4,9), (1,7)]
    
    if model == 'linear':
        subtype_lines = list()
        
        for s_ix in range(subtypes):
            pts = np.random.rand(F, 2) * 10
            
            # if increasing: have only increasing lines
            if increasing:
                pts = np.empty((F,2))
                for f_ix in range(F):
                    if extreme:
                        pt0 = np.random.rand() * 3
                        pt1 = np.random.rand() * 3 + 7
                    else:
                        pt0 = np.random.rand() * 10
                        pt1 = np.random.rand() * (10 - pt0) + pt0
                    pts[f_ix,0] = pt0
                    pts[f_ix,1] = pt1
            else:
                if extreme:
                    for f_ix in range(F):
                        pt0 = np.random.rand() * 3
                        pt1 = np.random.rand() * 3 + 7
                        if np.random.rand() > 0.5:
                            pts[f_ix,0] = pt0
                            pts[f_ix,1] = pt1
                        else:
                            pts[f_ix,0] = pt1
                            pts[f_ix,1] = pt0
                else:
                    pts = np.random.rand(F, 2) * 10
            true_lines = [np.poly1d(np.polyfit([0,4], pts[f_ix], 1)) for f_ix in range(F)]
            subtype_lines.append(true_lines)
        
        if use_godmode:
            subtype_lines = [[], [], []]
            # overwrite everything if in godmode
            for f_ix in range(F):
                godmode_ix = range(3)
                np.random.shuffle(list(godmode_ix))
                for s_ix in range(subtypes):
                    pt0 = godmode[s_ix][0] 
                    pt1 = godmode[s_ix][1]
                    
                    lfit = np.polyfit([0,4], [pt0, pt1], 1)
#                     import pdb; pdb.set_trace()
                    subtype_lines[s_ix].append(np.poly1d(lfit))
        if make_x:
            print('using make_X')
            subtype_lines = [[], []]
            # overwrite everything if in godmode
            for f_ix in range(F):
                for s_ix in range(subtypes):
                    pt0 = make_x_coord[s_ix][0] 
                    pt1 = make_x_coord[s_ix][1]
                    
                    lfit = np.polyfit([0,4], [pt0, pt1], 1)
#                     import pdb; pdb.set_trace()
                    subtype_lines[s_ix].append(np.poly1d(lfit))
        if case1:
            print('using case 1')
            subtype_lines = [[], []]
            # overwrite everything if in godmode
            for f_ix in range(F):
                for s_ix in range(subtypes):
                    pt0 = case1_coord[s_ix][0] 
                    pt1 = case1_coord[s_ix][1]
                    
                    lfit = np.polyfit([0,4], [pt0, pt1], 1)
#                     import pdb; pdb.set_trace()
                    subtype_lines[s_ix].append(np.poly1d(lfit))
            
        if case2:
            def flat_up(x):
                if x < 2:
                    return 5
                else:
                    z = np.poly1d(np.polyfit([2,4], [5, 9], 1))
                    return z(x)
            
            def flat_down(x):
                if x < 2:
                    return 5
                else:
                    z = np.poly1d(np.polyfit([2,4], [5, 0.5], 1))
                    return z(x)
                
            print('using case 2')
            subtype_lines = [[], []]
            for f_ix in range(F):
                for s_ix in range(subtypes):
                    if s_ix == 0:
                        subtype_truth_line = flat_up
                    if s_ix == 1:
                        subtype_truth_line = flat_down
                    subtype_lines[s_ix].append(subtype_truth_line)
                
        if case3:
            flat = lambda x: 5
            
            def down_up(x):
                if x < 1.2:
                    z = np.poly1d(np.polyfit([0,30], [9, 0.5], 1))
                    return z(x)
                if x >= 1.5 and x < 2.5:
                    return 0.5
                else:
                    z = np.poly1d(np.polyfit([2.5,4], [0.5, 8.5], 1))
                    return z(x)
            
            print('using case 3')
            subtype_lines = [[], []]
            for f_ix in range(F):
                for s_ix in range(subtypes):
                    if s_ix == 0:
                        subtype_truth_line = flat
                    if s_ix == 1:
                        subtype_truth_line = down_up
                    subtype_lines[s_ix].append(subtype_truth_line)
                    
        if case4 or case5:
#             def down_up(x):
#                 if x < 30:
#                     z = np.poly1d(np.polyfit([0,30], [9, 0.5], 1))
#                     return z(x)
#                 if x >= 30 and x < 62:
#                     return 0.5
#                 else:
#                     z = np.poly1d(np.polyfit([2.5,4], [0.5, 8.5], 1))
#                     return z(x)
            
            print('using case 4 or 5')
            subtype_lines = [[], []]
            for f_ix in range(F):
                for s_ix in range(subtypes):
                    subtype_truth_line = np.poly1d([2.3, -8.9, 9.1])
                    subtype_lines[s_ix].append(subtype_truth_line)
                    
        if case6:
            """
            One is flat, one is quadratic
            if gap<0., subtypes overlap.
            """
            flat = lambda x: -gap
            
#             print('using case 6')
            subtype_lines = [[], []]
            for f_ix in range(F):
                for s_ix in range(subtypes):
                    if s_ix == 0:
                        subtype_truth_line = flat
                    if s_ix == 1:
                        a,b,c = 0.25, -2.2, 5
                        subtype_truth_line = np.poly1d([a, b, c])
                    subtype_lines[s_ix].append(subtype_truth_line)
        
        if case7:
            """
            One is linearly increasing, one is quadratic
            if overlap=True, subtypes overlap.
            """
            subtype_lines = [[], []]
            for f_ix in range(F):
                for s_ix in range(subtypes):
                    if s_ix == 0:
                        ref_line_pts = [[0.4, -5], [0.4, 0]]
                        overlap_idx  = 1 if overlap else 0
                        
                        subtype_truth_line = np.poly1d(ref_line_pts[overlap_idx])
                    if s_ix == 1:
                        a,b,c = 0.25, -2.2, 5
                        subtype_truth_line = np.poly1d([a, b, c])
                    subtype_lines[s_ix].append(subtype_truth_line)
                    
        if case8:
            """
            Both are quadratic in opposite directions
            if gap < 0., they overlap
            """
#             print('using case 6')
            subtype_lines = [[], []]
            for f_ix in range(F):
                for s_ix in range(subtypes):
                    if s_ix == 0:
                        a,b,c = 0.25, -2.2, 5
                        subtype_truth_line = np.poly1d([a, b, c+gap])
                    if s_ix == 1:
                        a,b,c = -0.25, 2.2, -5
                        subtype_truth_line = np.poly1d([a, b, c])
                    subtype_lines[s_ix].append(subtype_truth_line)
        if case9:
            """
            Both are quadratic in same direction but one is more flat
            """
#             print('using case 6')
            subtype_lines = [[], []]
            for f_ix in range(F):
                for s_ix in range(subtypes):
                    if s_ix == 0:
                        a,b,c = 0.25, -2.2, 5
                        subtype_truth_line = np.poly1d([a, b, c+gap])
                    if s_ix == 1:
                        a,b,c = -0.25, 2.2, -5
                        subtype_truth_line = np.poly1d([a, b, c])
                    subtype_lines[s_ix].append(subtype_truth_line)
        if case10:
            """
            Both are quadratic in same direction but one is more flat
            """
#             print('using case 6')
            subtype_lines = [[], []]
            for f_ix in range(F):
                for s_ix in range(subtypes):
                    if s_ix == 0:
                        theta = [-0.07, 0.95,-3,2.7]
                        subtype_truth_line = np.poly1d(theta)
                    if s_ix == 1:
                        theta = [-0.07, 0.95,-3,-10]
                        subtype_truth_line = np.poly1d(theta)
                    subtype_lines[s_ix].append(subtype_truth_line)
            
    elif model == 'none':
        true_centers = np.empty((subtypes, F))
        hardcode = [1,3,7]
        for s_ix in range(subtypes):
            for f_ix in range(F):
                true_centers[s_ix, f_ix] = hardcode[s_ix]

    
    # generate patient data
    
    data = np.empty((N,M,F))
    patient_subtypes = np.empty((N,M))
    patient_times = np.empty((N,M))
    patient_ids = np.empty((N,M))
    patient_obs_times = np.empty((N,M))
    
    
    for p_id in range(N):
        # draw subtype randomly
        p_subtype = np.random.choice(subtypes)
        
        # if case5, start is either 5 or 15
        if case5:
            delta = 0.2 if p_subtype == 1 else 1.5
            offsets = np.random.rand(M-1) * 4
            all_times = np.zeros(M)
            all_times[1:] = offsets + delta
            all_times[0] = delta
            
        else:
            # OLD
            all_times = np.random.uniform(0,10,size=M)                
            
            # NEW time method
#             first_time = np.random.uniform(0, 3)
            
#             all_times = np.zeros(M)
#             all_times[0] = first_time
#             for i in range(1, M):
#                 all_times[i] = np.random.uniform(first_time, 10)
            
            
        all_times = sorted(all_times)
        for visit_id in range(M):
            patient_subtypes[p_id,visit_id] = p_subtype
            patient_ids[p_id,visit_id] = p_id
            
            t = all_times[visit_id]
            
            if model == 'linear':
                patient_times[p_id, visit_id] = t
                p_true_line = subtype_lines[p_subtype]

                p_obs = np.empty(F)
                for f_ix in range(F):
                    p_obs = p_true_line[f_ix](t) + np.random.normal(scale=noise)
                    try:
                        data[p_id, visit_id,f_ix] = p_obs
                    except:
                        import pdb; pdb.set_trace()
                earliest_patient_time = patient_times[p_id].min()
                
                try:
                    patient_obs_times[p_id] = patient_times[p_id] - earliest_patient_time
                except:
                    import pdb; pdb.set_trace()
            
            elif model == 'none':
                patient_times[p_id, visit_id] = t
                p_obs = np.empty(F)
                for f_ix in range(F):
                    p_obs = np.random.normal(loc=true_centers[p_subtype,f_ix])
                    data[p_id, visit_id,f_ix] = p_obs
                
    df_data = np.reshape(data, (N*M, F))
    df_subtypes = np.reshape(patient_subtypes, (N*M, 1))
    df_times = np.reshape(patient_times, (N*M, 1))
    df_ids = np.reshape(patient_ids, (N*M, 1))
    df_obs_times = np.reshape(patient_obs_times, (N*M, 1))
    
#     df_data = np.clip(df_data, 0.1, df_data.max())
    
    df = pd.DataFrame(df_data)
    df['subtype'] = df_subtypes
    df['time'] = df_times
    df['p_id'] = df_ids
    if obs_time:
        df['obs_time'] = df_obs_times
    
    if model == 'linear':
        return df, subtype_lines
    elif model == 'none':
        return df, true_centers

def load_piecewise_synthetic_data(subtypes=3, increasing=False, 
                        D=1, N=500,M=4, noise=0.5, N_pts=5):
    """
    returns dataframe with columns subtype, p_id, time, obs_time, numbered columns
    
    increasing (bool): is data monotonically increasing per subtype?
    N_pts (int): how many points define each subtype progression?
    """
    
    x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
    y = np.sin(x)
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.arange(0, 2*np.pi, np.pi/50)
    ynew = interpolate.splev(xnew, tck, der=0)
    
    
    
    data = np.empty((N,M,D))
    patient_subtypes = np.empty((N,M))
    patient_times = np.empty((N,M))
    patient_ids = np.empty((N,M))
    patient_obs_times = np.empty((N,M))
    
    
    # for each dimension, pick some random points x in [0,10] and y in [0,1] to determine data 
    subtype_pts         = np.random.random((subtypes,D,N_pts,2))
    subtype_pts[:,:,:,0] *= 10
    
    if increasing:
        for d in range(D):
            for k in range(subtypes):
                subtype_pts[k,d,:,1] = sorted(subtype_pts[k,d,:,1])
                subtype_pts[k,d,:,0] = sorted(subtype_pts[k,d,:,0])
        
    # interpolate between these points for each subtype for each dimension
    
    interpolate_tck = dict()
    
    for p_id in range(N):
        
        # draw subtype randomly
        p_subtype = np.random.choice(subtypes)
        k         = p_subtype
        
        all_times = np.random.uniform(0,10,size=M)
        all_times = sorted(all_times)
        
        for d in range(D):
            if (p_subtype, d) in interpolate_tck:
                tck = interpolate_tck[(p_subtype,d)]
            else:
                sort_idx = np.argsort(subtype_pts[k,d,:,0])
                xvals = subtype_pts[k,d,:,0][sort_idx]
                yvals = subtype_pts[k,d,:,1][sort_idx]
                tck = interpolate.splrep(xvals, yvals, s=0)
                interpolate_tck[(p_subtype,d)] = tck
            yvals = interpolate.splev(all_times, tck, der=0)
                
            for visit_id in range(M):
                patient_subtypes[p_id,visit_id] = p_subtype
                patient_ids[p_id,visit_id] = p_id
                t     = all_times[visit_id]
                p_obs = np.random.normal(loc=yvals[visit_id],scale=noise)
                
                patient_times[p_id, visit_id]    = t
                data[p_id,visit_id,d]            = p_obs
                patient_obs_times[p_id,visit_id] = t - min(all_times)
        # draw points 
        
    
    df_data = np.reshape(data, (N*M, D))
    df_subtypes = np.reshape(patient_subtypes, (N*M, 1))
    df_times = np.reshape(patient_times, (N*M, 1))
    df_ids = np.reshape(patient_ids, (N*M, 1))
    df_obs_times = np.reshape(patient_obs_times, (N*M, 1))
    
#     df_data = np.clip(df_data, 0.1, df_data.max())
    
    df = pd.DataFrame(df_data)
    df['subtype'] = df_subtypes
    df['time'] = df_times
    df['p_id'] = df_ids
#     if obs_time:
    df['obs_time'] = df_obs_times
    return df, subtype_pts

if __name__ == '__main__':
    data = load_sigmoid_data(subtypes=2, F=3, N=1000, M=17, noise=0.5, ppmi=True)