import argparse
import csv
import numpy as np

import os.path
from os import path
import pandas as pd
pd.options.mode.chained_assignment = None
import sys

# from model_utils import get_loader, change_size

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

def load_ppmi_baseline_data(p_ids, use_healthy=True):
#     if not os.path.exists('/afs/REDACTED.REDACTED.edu/group/REDACTED/datasets/ppmi/visit_feature_inputs_asof_2019Jan24_using_CMEDTM/'):
#         print('Error: PPMI path does not exist. Are you on the wrong machine? PPMI experiments can only be run on Chronos/Kratos.')
#         return 
    
    ppmi_healthy_csv = '~/chf-github/data/HC_baseline.csv'
    ppmi_csv = '~/chf-github/data/PD_baseline.csv'
    
    ppmi = pd.read_csv(ppmi_csv)
    healthy = pd.read_csv(ppmi_healthy_csv)
    
    baseline_cols = [
    'MALE',
    'HISPLAT',
    'RAWHITE',
    'RAASIAN',
    'RABLACK',
    'RAINDALS',
    'RAHAWOPI',
    'RANOS',
    'BIOMOMPD',
    'BIODADPD',
    'FULSIBPD',
    'HAFSIBPD',
    'MAGPARPD',
    'PAGPARPD',
    'MATAUPD',
    'PATAUPD',
    'KIDSPD',
    'EDUCYRS',
    'RIGHT_HANDED',
    'LEFT_HANDED',
    'UPSITBK1',
    'UPSITBK2',
    'UPSITBK3',
    'UPSITBK4',
    'UPSIT'
    ]
    if use_healthy:
        df = pd.concat([healthy, ppmi])
    else:
        df = ppmi
    
    if (df['PATNO'].values == len(p_ids)).all():
        return df[baseline_cols]
    else:
        df.index = df['PATNO']
        int_pids = [int(i) for i in p_ids]
        df2 = df.reindex(int_pids)
        return df2[baseline_cols]
    
def load_ppmi_data(use_healthy=True, add_time_offset=False):
    """
    data is explored in ExploreData.ipynb
    """
    print('loading PPMI data')
#     if not os.path.exists('/afs/REDACTED.REDACTED.edu/group/REDACTED/datasets/ppmi/visit_feature_inputs_asof_2019Jan24_using_CMEDTM/'):
#         print('Error: PPMI path does not exist. Are you on the wrong machine? PPMI experiments can only be run on Chronos/Kratos.')
#         return 
    
    ppmi_healthy_csv = '~/chf-github/data/HC_totals_across_time.csv'
    ppmi_csv = '~/chf-github/data/PD_totals_across_time.csv'
    
    ppmi = pd.read_csv(ppmi_csv)
    healthy = pd.read_csv(ppmi_healthy_csv)

#     print('Num healthy patients: %d' % len(healthy['PATNO'].unique()))
#     print('Num PPMI patients: %d' % len(ppmi['PATNO'].unique()))
    
    """
    max of (‘NUPDRS3_on’, ‘NUPDRS3_off’, and ‘NUPDRS3_maob’)
    ‘MOCA’ is the most common cognitive assessment
    ‘SCOPA-AUT’ for autonomic
    ‘NUPDRS1’ (part 1 of MDS-UPDRS exam) for non-motor symptoms in general
    """

    cols = ['MOCA', 'SCOPA-AUT', 'NUPDRS1', 'NUPDRS3_on', 'NUPDRS3_off', 'NUPDRS3_maob', 'PATNO', 'EVENT_ID_DUR']
    
    healthy_df = healthy[cols]
    ppmi_df = ppmi[cols]
    
#     print('Num healthy patients: %d' % len(healthy_df['PATNO'].unique()))
#     print('Num PPMI patients: %d' % len(ppmi_df['PATNO'].unique()))
    
    # Set subtypes
    healthy_df.loc[:,'subtype'] = 0
    ppmi_df.loc[:,'subtype'] = 1

    healthy_df['NUPRDRS3_agg'] = healthy['NUPDRS3_untreated']
    ppmi_df['NUPRDRS3_agg'] = ppmi_df.apply(lambda x: max([x['NUPDRS3_on'], x['NUPDRS3_off']]), axis=1)
    
    # Concatenate into one dataframe
    if use_healthy:
#         print('Using healthy controls too')
        df = pd.concat([healthy_df, ppmi_df])
    else:
        df = ppmi_df
    
#     df['NUPRDRS3_agg'] = df.apply(lambda x: max([x['NUPDRS3_on'], x['NUPDRS3_off']]), axis=1)
    df = df.drop(['NUPDRS3_on', 'NUPDRS3_off', 'NUPDRS3_maob'], axis=1)
    
    # scale all cols between 0 and 1
    cols_with_max_larger_than_1 = df.columns[df.max() > 1.]

    cols_with_max_larger_than_1 = [i for i in cols_with_max_larger_than_1 if i not in ['PATNO', 'EVENT_ID_DUR']]
    for col in cols_with_max_larger_than_1:
        df[col] = df[col] / df[col].max()
        
    if add_time_offset:
        df['obs_time'] = df['EVENT_ID_DUR']
        df = df.sort_values(['PATNO', 'EVENT_ID_DUR'])
        
        for patno in df['PATNO'].unique():
            if np.random.random() < add_time_offset:
                num_visits = len(df[df['PATNO'] == patno])
                patno_index = df[df['PATNO'] == patno].index
                for index_to_drop in patno_index[:int(num_visits * 0.2)]: 
                    df = df.drop(index_to_drop, axis=0)
        earliest_time = df.groupby('PATNO').min()['EVENT_ID_DUR']
        df['obs_time'] = df.apply(lambda x: x['obs_time'] - earliest_time.loc[x['PATNO']], axis=1)
        
        print('Dropped first few visits from %d patients' % (add_time_offset * 100))        
        
    else:
        df['obs_time'] = df['EVENT_ID_DUR']
    
    end_cols = ['subtype', 'EVENT_ID_DUR', 'PATNO', 'obs_time']
    feat_cols = [i for i in df.columns if i not in end_cols]
    cols = feat_cols + end_cols

    return df[cols]

def load_mmrf_data(args):
    if not os.path.exists('/afs/REDACTED.REDACTED.edu/group/REDACTED/users/REDACTED/trvae/'):
        print('Error: MMRF path does not exist. Are you on the wrong machine? MMRF experiments can only be run on Chronos/Kratos.')
        return 
    else:
        sys.path.append('/afs/REDACTED.REDACTED.edu/group/REDACTED/users/REDACTED/trvae/')
        from data.mm_dataset import load_mmrf
    
    folds = range(5)
    mmdata = load_mmrf(fold_span = folds, digitize_K = 20, digitize_method = 'uniform',include_mtype=True)
    
    feature_names = ['chem_bun', 'chem_creatinine', 'chem_albumin', 'serum_kappa',
      'serum_m_protein', 'serum_iga', 'serum_igg', 'serum_igm',
      'serum_lambda']
    feature_idx = np.array([2,4,1,7,8,12,13,14,15])
    
    # Step 1: get features in feature_names from x (num_patients x num_timesteps x num_features)
    mmrf = np.concatenate([mmdata[1]['train']['x'][:,:,feature_idx],
                           mmdata[1]['valid']['x'][:,:,feature_idx],
                           mmdata[1]['test']['x'][:,:,feature_idx]
                          ])

    # Step 2: filter for people who are on line1 (num_patients x num_timesteps x 1)
    line1_idx = np.where(mmdata[1]['train']['feature_names_a'] == ['line1'])[0]
    is_line1 = np.concatenate([mmdata[1]['train']['a'][:,:,line1_idx] == 1, 
                               mmdata[1]['valid']['a'][:,:,line1_idx] == 1,
                               mmdata[1]['test']['a'][:,:,line1_idx] == 1])
    is_line1_mask = np.tile(is_line1, (1,1,len(feature_idx)))
    mmrf[~is_line1_mask] = np.nan
    
    # Step 3: Add nan values back in from censored
    is_censored = np.concatenate([mmdata[1]['train']['m'][:,:,feature_idx] == 1, 
                               mmdata[1]['valid']['m'][:,:,feature_idx] == 1,
                               mmdata[1]['test']['m'][:,:,feature_idx] == 1])
    is_null_mask = ~is_censored
    mmrf[is_null_mask] = np.nan
    
    # Step 4: Are there any time vals that are entirely empty?
    drop_cols = np.isnan(mmrf).all(axis=0).all(axis=1)
    mmrf = mmrf[:,~drop_cols,:]
    
    # Step 5: determine heavy chain and light chain
    is_heavy = np.concatenate([mmdata[1]['train']['b'][:,10],
                              mmdata[1]['valid']['b'][:,10],
                              mmdata[1]['test']['b'][:,10]])
    
    # Step 6: Scale everything between 0 and 1
    max_vals_by_col = np.nanmax(np.nanmax(mmrf, axis=0), axis=0)
    
    for col_i in range(len(feature_names)):
        mmrf[:,:,col_i] = mmrf[:,:,col_i] / max_vals_by_col[col_i]
        
    mmrf[np.isnan(mmrf)] = -1000
    
    print('Num patients: %d' % len(is_heavy))
    print('Num heavy chain: %d' % np.sum(is_heavy))
    
    s_collect = np.array(is_heavy).astype('float32')
    Y_collect = mmrf.astype('float32')
    
    assert (mmdata[1]['test']['a'][0,~drop_cols,0].shape == mmdata[1]['train']['a'][0,~drop_cols,0].shape)
    assert (mmdata[1]['test']['a'][0,~drop_cols,0].shape == mmdata[1]['valid']['a'][0,~drop_cols,0].shape)
    
    t_collect = mmdata[1]['train']['a'][0,~drop_cols,0]
    
    N = mmrf.shape[0]
    t_collect = change_size(t_collect, [0,2], [N,1,1]).astype('float32')
    
    collect_dict = {
        'Y_collect': Y_collect,
        't_collect': t_collect,
        's_collect': s_collect,
        'obs_t_collect': t_collect
    }
    
    data_loader = get_loader(Y_collect, s_collect, t_collect)
    return data_loader, collect_dict, feature_names

def load_mmrf_baseline_data(args):
    return

def load_chf_data(only_echoes=True, only_pef=False):
    """
    Data csvs are created in RunCHFData.ipynb
    """
    if not os.path.exists('/home/REDACTED/chf-github/vae/'):
        print('Error: CHF path does not exist. Are you on the wrong machine? CHF experiments can only be run on r730/k80/etc.')
        return 
    else:
        sys.path.append('/home/REDACTED/chf-github/vae/')
        sys.path.append('/home/REDACTED/chf/scripts')
        sys.path.append('/home/REDACTED/chf/')
        from echo_mortality import get_echo_dirty, get_clean_echo, get_echo_cols, expand_diagnosis
    
    # See RunCHFData.ipynb for code to make data/chf_sublign_data.csv
    if only_echoes:
        print('Using only CHF echo values')
        if os.path.exists('../clinical_data/chf_sublign_data_echo_only.csv'):
            chf = pd.read_csv('../clinical_data/chf_sublign_data_echo_only.csv')
        else:
            chf = pd.read_csv('clinical_data/chf_sublign_data_echo_only.csv')
        
        if only_pef:
            print('Only using HFpEF patients by first echo')
            chfcopy = chf.copy()
            chfcopy = chfcopy.sort_values(['PatientID', 'time'])
            byPatientID = chfcopy.groupby('PatientID',as_index=False).first()
            # renormalized values so Ejection Fraction < 5 means HFpEF
            byPatientID_pef = byPatientID[byPatientID['Left Ventricle - Ejection Fraction'] < 5.]
            pef_pids = set(byPatientID_pef['PatientID'].values)
            chf = chf[chf['PatientID'].apply(lambda x: x in pef_pids)]
    else:
        try:
            chf = pd.read_csv('../clinical_data/chf_sublign_data.csv')
        except:
            chf = pd.read_csv('clinical_data/chf_sublign_data.csv')
    
    # for each p_id, subtract from earliest such that each patient starts at t=0 as the earliest
    earliest_times = chf.groupby('PatientID')['time'].min()

    earliest_times_dict = dict()

    # find earliest time for PatientID
    for p, val in earliest_times.iteritems():
        earliest_times_dict[p] = val
        zero_time = list()

    for index, row in chf.iterrows():
        p_id = row['PatientID']
        time = row['time']
        earliest_time = earliest_times_dict[p_id]

        zero_adj_time = time - earliest_time
        zero_time.append(zero_adj_time)

    chf['zero_time'] = zero_time

    chf['time'] = chf['zero_time']
    chf = chf.drop(['zero_time'],axis=1)
    
    # scale all data so features are between 0 and 1
    cols_with_max_larger_than_1 = chf.columns[chf.max() > 1.]

    cols_with_max_larger_than_1 = [i for i in cols_with_max_larger_than_1 if i not in ['time', 'PatientID']]
    for col in cols_with_max_larger_than_1:
        chf[col] = chf[col] / chf[col].max()
        
    # Remove patients with only 1 visit (~300)
    all_patients = chf.groupby('PatientID')['time'].count().index
    patients_with_only_1_visit = set((all_patients[chf.groupby('PatientID')['time'].count() == 1].values))
    chf = chf[chf['PatientID'].apply(lambda x: x not in patients_with_only_1_visit)]
    
    sublign_cols = [col for col in chf.columns if col not in ['subtype', 'obs_time','time', 'PatientID']] + ['subtype', 'time', 'PatientID', 'obs_time']
    
    print('Num patients: %d' % len(chf['PatientID'].unique()))
    print('Avg visits per patient: %.1f' % chf.groupby('PatientID').count()['subtype'].mean())
    print('Max visits for a patient: %.1f ' % chf.groupby('PatientID').count()['subtype'].max())
    print('Min visits for a patient: %.1f ' % chf.groupby('PatientID').count()['subtype'].min())
    
    print('10-percentile visits: %d' % np.percentile(chf.groupby('PatientID').count()['subtype'], 10))
    print('50-percentile visits: %d' % np.percentile(chf.groupby('PatientID').count()['subtype'], 50))
    print('80-percentile visits: %d' % np.percentile(chf.groupby('PatientID').count()['subtype'], 80))
    
    chf['obs_time'] = chf['time']
    return chf[sublign_cols]
    
def load_chf_baseline_data(p_ids):
    if not os.path.exists('/home/REDACTED/chf-github/vae/'):
        print('Error: CHF path does not exist. Are you on the wrong machine? CHF experiments can only be run on r730/k80/etc.')
        return 
    else:
        sys.path.append('/home/REDACTED/chf-github/vae/')
        sys.path.append('/home/REDACTED/chf/scripts')
        sys.path.append('/home/REDACTED/chf/parsing')
        sys.path.append('/home/REDACTED/chf/')
        from echo_mortality import get_echo_dirty, get_clean_echo, get_echo_cols, expand_diagnosis
        from chf_labs import get_patients, get_echo
        from chf_data_utils import get_race_group, shorted_names, expand_diagnosis
        
    patients = get_patients()
    patients['is_female'] = (patients['Sex'] == 'F').apply(int)
    patients['race_group'] = patients['Race'].apply(get_race_group)
    dummies = pd.get_dummies(patients['race_group'], prefix='race')
    patients[dummies.columns] = dummies
    
    patients = expand_diagnosis(patients, diag_num=20)
    
    echo = get_echo()
    echo['Weight (lb)'] = echo.mask(echo['Weight (lb)'] > 580)['Weight (lb)']
    echo['Height'] = echo.mask(echo['Height'] > 78)['Height']

    echo['BMI'] = 703 * echo['Weight (lb)'] / (echo['Height'] * echo['Height'])
    echo['Obesity'] = (echo['BMI'] > 30).apply(int)
    
    echo_baseline = echo[['patient', 'BMI', 'Obesity']].groupby('patient', as_index=False).max()[['patient', 'BMI', 'Obesity']]
    
    echo_baseline['PatientID'] = echo_baseline['patient']
    
    ignore_cols = ['Date', 'Dispo', 'ED.CPT', 'ED.LengthOfStay', 'ED.triageAcuity','Ethnicity', 'Expired',
              'Race','Sex','visit_id', 'race_group', 'patient', 'PatientID'
              ]
    patients_unique = patients.groupby('PatientID').max()
    df = patients_unique.merge(echo_baseline, how='right', on='PatientID')
    feat_cols = [i for i in df.columns if i not in ignore_cols]
    
    df2 = df.copy()
    
    if (df['PatientID'].values == len(p_ids)).all():
        return df[feat_cols]
    else:
        df2.index = df2['PatientID']
        int_pids = [int(i) for i in p_ids]
        df3 = df2.reindex(int_pids)
        
        return df3[feat_cols]
    
    
    # TODO: make sure in the same order as patient ID input
    print('WARNING: have no order CHF baseline data by patient IDs ')
    return df[feat_cols]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    args = parser.parse_args()
    
    load_mmrf_data(args)
    