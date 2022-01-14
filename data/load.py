from clinical_loader import load_ppmi_data, load_chf_data
from synthetic_data import load_synthetic_data, load_sigmoid_data

def load_data_format_by_num(data_format_num):
    if data_format_num == 1:
        data = load_sigmoid_data(subtypes=2, F=3, N=1000, M=4, noise=0.25)
    elif data_format_num == 2:
        data = load_sigmoid_data(subtypes=2, F=3, N=4000, M=1, noise=0.25)
    elif data_format_num == 3:
        data, _ = load_synthetic_data(subtypes=2, F=1, N=1000, M=4, gap=-2, case6=True,noise=1.25)
    elif data_format_num == 4:
        data, _ = load_synthetic_data(subtypes=2, F=1, N=1000, M=4, gap=2, case6=True,noise=1.25)
    elif data_format_num == 5:
        data, _ = load_synthetic_data(subtypes=2, F=1, N=1000, M=4, overlap=True, case7=True,noise=1.25)
    elif data_format_num == 6:
        data, _ = load_synthetic_data(subtypes=2, F=1, N=1000, M=4, overlap=False, case7=True,noise=1.25)
    elif data_format_num == 7:
        data, _ = load_synthetic_data(subtypes=2, F=1, N=1000, M=4, gap=-2, case8=True,noise=0.25)
    elif data_format_num == 8:
        data, _ = load_synthetic_data(subtypes=2, F=1, N=1000, M=5, gap=2, case8=True,noise=0.25)
    elif data_format_num == 10:
        data, _ = load_synthetic_data(subtypes=2, F=1, N=1000, M=5, gap=2, case10=True,noise=0.25)
    elif data_format_num == 11:
        data = load_sigmoid_data(subtypes=2, F=3, N=1000, M=17, noise=0.5, ppmi=True,missing_rate=0.5)
    elif data_format_num == 12:
        data = load_sigmoid_data(subtypes=2, F=3, N=1000, M=17, noise=0.5, ppmi=True,missing_rate=0.25)
    elif data_format_num == 13:
        data = load_sigmoid_data(subtypes=2, F=3, N=1000, M=17, noise=0.5, ppmi=True,missing_rate=0.0)
    elif data_format_num == 14:
        data = load_sigmoid_data(subtypes=2, F=3, N=1000, M=17, noise=0.5, ppmi=True, missing_rate=0.10)
    elif data_format_num == 15:
        data = load_sigmoid_data(subtypes=2, F=3, N=1000, M=17, noise=1., ppmi=True, missing_rate=0.0)
    return data

def load_data_format(data_format_num,trial_num=None, cache=True):
    """
    For use in illustrative experiments
    
    cache: if cache version exists, load
    
    Data 1: Sigmoid (1000 patients, 4 visits, 3 dimensions) - sigmoid
    Data 2: Sigmoid (4000 patients, 1 visits, 3 dimensions) - sigmoid
    Data 3: Quadratic-overlap (1000 patients, 4 visits, 3 dimensions) - quad
    Data 4: Quadratic-gap (1000 patients, 4 visits, 3 dimensions) - quad
    Data 5: Quadratic-linear-overlap (1000 patients, 4 visits, 3 dimensions) - quad-linear 
    Data 6: Quadratic-linear-gap (1000 patients, 4 visits, 3 dimensions) - quad-linear 
    Data 7: Quadratic-quad-overlap (1000 patients, 4 visits, 3 dimensions) - quad-quad 
    Data 8: Quadratic-quad-gap (1000 patients, 4 visits, 3 dimensions) - quad-quad
    Data 11: Synthetic data with missingness rates to match PPMI dataset (M=17, D=4, missing between 0.7 and 0.3)
    """
    
    # if no trial num, load from scratch
    if trial_num == None:
        print('loading from scratch')
        data = load_data_format_by_num(data_format_num)
    else:
        import os
        import pickle
        
        fname = '/home/REDACTED/chf-github/trial_data/data%d_trial%d.pk' % (data_format_num, trial_num)
        
        if os.path.exists(fname) and cache:
            data = pickle.load(open(fname, 'rb'))
        else:
            fname = 'data%d_trial%d.pk' % (data_format_num, trial_num)
            data = load_data_format_by_num(data_format_num)
            pickle.dump(data, open(fname, 'wb'))
    return data

def sigmoid():
    data = load_sigmoid_data(subtypes=2, F=3, N=1000, M=4, noise=0.25)
    return data

def quadratic(gap=-2):
    data, _ = load_synthetic_data(subtypes=2, F=1, N=1000, M=4, gap=gap, case6=True)
    return data

def chf():
    data = load_chf_data()
    return data

def parkinsons():
    data = load_ppmi_data()
    return data

if __name__=='__main__':
    dataset = parkinsons()
    import pdb;pdb.set_trace()