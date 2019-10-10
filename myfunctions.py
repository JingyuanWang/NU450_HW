'''
# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose: 
# define my functions
#     1. bootstrap
#     2. gen_poly
# ------------------------------------------------------------------------
'''

import numpy as np
import pandas as pd
from sklearn.utils import resample
import scipy.optimize as opt
import itertools as iter

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


# generate polynomials
def gen_poly(df_input, varnames, poly_max):
    '''given a dataframe and a list of variables:
    generate polynomials of the variables (up to (poly_max)th order) and save to the dataframe
    
    return: the new dataframe and a list of polynomial variable names'''
    
    df = df_input.copy()
    n = len(varnames)
    
    # specify group name: all the variable names start with groupname. 
    # example: poly_var1_var2_1_3 = var1^{1} \times var2^{3}
    groupname = 'poly'
    for i in range(n):
        groupname = groupname + '_' + varnames[i]
    
    
    # STEP 0 constant:
    polynames = '_0'*n
    new_polyvar_name = groupname + polynames
    df[new_polyvar_name] = 1
    print('-- generate: {} --'.format(new_polyvar_name))
    
    # STEP 1 just 1 variable, 1st order-highest order: variable it self
    for i in range(n):
        for j in range(1,poly_max+1): 
            polynames = '_0'*(i) + '_' + str(j) + '_0' *(n-i-1)
            new_polyvar_name = groupname + polynames
            df[new_polyvar_name] = df[varnames[i]].values**poly_max
            print('-- generate: {} --'.format(new_polyvar_name))
    
    # STEP 2 poly 2-poly_max, with number of variables >=2
    lenth = df.shape[0]

    for poly_max in range(2,poly_max+1):
        for n_include in range(2,n+1):
            # (1) pick n_include variables from all n variables
            for comb_of_vars in (iter.combinations(range(0,n), n_include)):
                varnames_include = [varnames[comb_of_vars[0]]]
                for i in range(1,n_include):
                    varnames_include = varnames_include + [varnames[comb_of_vars[i]]]
                print('-- {}'.format(varnames_include))
            
                # (2) give each included variable a power
                cumul_list = list(iter.combinations(range(1,poly_max), n_include-1))
                # For number of variables == n_include, total number of unique combinations = len(cumul_list)
                # each combination represents one new polynomial variable, generate them 1-by-1
                for i in range(len(cumul_list)):
                
                    # start to generate 1 new poly var
                    cumul = list(cumul_list[i])
                    power = list(cumul)
                    power[1:] = [cumul[i + 1] - cumul[i] for i in range(len(cumul)-1)]
                    power.append(poly_max-cumul[-1])
                
                    output = np.zeros(lenth)
                    new_var_name = ''
                    polynames = ''
                    for var in range(n_include):
                        output = output + df[varnames_include[var]].values**power[var]
                        new_var_name = new_var_name + varnames_include[var] + '_'
                        polynames = polynames + '_' + str(power[var])
                    
                    new_polyvar_name = 'poly_' + new_var_name[:-1] + polynames
                    df[new_polyvar_name] = output
                    print('-- generate: {} --'.format(new_polyvar_name))
                    # end of generating 1 new poly var
    
    print('# --- Finish generating polynomials ---')
    polynames = [n for n in df.columns.values if n.startswith('poly')]
    
    return [df, polynames]


