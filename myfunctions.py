'''
# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose: 
# define my functions
#     1. bootstrap
#
# ------------------------------------------------------------------------
'''

import numpy as np
import pandas as pd
from sklearn.utils import resample
import scipy.optimize as opt

def bootstrap(obj, arguments, beta_hat, beta_initial = None, subsample_size = None, total_times = None):
    '''This function returns [std_err,var_cov] 
    
    Input: 
    -- obj: objective function for the minimization
    -- arguments: the arguments plut into the objective function (except beta), 
                  by default the first one should be a dataframe which is the analysis sample.
                  This should be exact the same as the arguments in the main optimization.
    -- beta_hat: results from the main optimization. 
                 We need it here just to get its length (number of parameters).
    
    Optional Input:
    -- beta_initial: beta_initial for sub-sample optimizations. Default: ones
    -- subsample_size: Default: ceil(sample_size * 0.8)
    -- total_times: total times of sub-sampling and optimization. Default: 200
    '''
    
    # STEP 0 clean the input -------------------------------------
    # (1) 
    # the first argument should be df, but we want to replace this to some sub-sample
    df = arguments[0]
    arguments = arguments[1:]
    
    # Optional input default values 
    # (2) beta_initial for sub-sample optimizations
    if beta_initial == None:
        beta_initial = np.ones(len(beta_hat))
        
    # (3) subsample_size
    if subsample_size == None:
        sample_size = df.shape[0]
        subsample_size = np.ceil(sample_size * 0.8).astype(int)
        # 0.8 has no theoretical support! I just randomly come up with this number. Check 480-3 notes for optimal!
    
    # (4) total_times
    if total_times == None:
        total_times = 200
    
    # STEP 1 run the optimization multiple times  -----------------
    beta = []
    for n in range(0,total_times):
        # resample
        index = df.index.values
        sample_index = resample(index, replace = True, n_samples=subsample_size)
        # optimize
        results = opt.minimize(obj,beta_initial,args = (df.loc[sample_index],) + arguments )
        # save beta
        beta.append(results.x)
        # print progress every 10 optimizations
        if n%20 == 0:
            print('--{}------'.format(n))
            print('beta: {}'.format(results.x))
            
    # get variance of beta
    b = np.column_stack(beta)
    var_cov = np.cov(b)
    std_err = np.sqrt(np.diag(var_cov))
    return [std_err,var_cov]