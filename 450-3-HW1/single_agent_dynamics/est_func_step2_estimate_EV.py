'''
# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose:
# define class: 
#
# ------------------------------------------------------------------------
'''

# Notations:
# M: num of markets
# F: num of firms in each market
# n_sample: number of simulation draws to integral

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt
from scipy.stats import norm
import importlib
from sklearn.utils import resample
import os,sys,inspect
import seaborn as sns
import matplotlib.pyplot as plt

# my functions
dirpath = os.getcwd()
i = 0
while(os.path.basename(dirpath) != "GitHub") and (i<=10):
    dirpath = os.path.dirname(dirpath)
    i = i + 1
targetdir = dirpath + '/tools'
if targetdir not in sys.path:
    sys.path.insert(0,targetdir)

from general import data_format
importlib.reload(data_format)

# ---------------------------------------------------------------------------------
# estimate EV: fixed point 
# ---------------------------------------------------------------------------------

def find_fixed_point(u, transition, EV0 = None , beta = 0.95, tol = 1e-8, maxiter = 1000):

    gamma = 0.577216
    num_of_states = len(u['0'])
    if EV0 is None:
        EV0 = {}
        EV0['0'] = np.ones(num_of_states)
        EV0['1'] = np.ones(num_of_states)
 
    # iteration : set up
    i = 0
    EV = EV0.copy()
    EV_new = {}
    diff = 9999
    
    # iteration: start
    while (diff > tol) and (i < maxiter):
        
        # 1. utility of each actions
        delta_0 = u['0'] + beta* EV['0']
        delta_1 = u['1'] + beta* EV['1']
        ### correction for exp^delta = inf. 
        ### if delta > 1000, we have exp1000 = inf. But we we need is ln(expA + expB). The final one is not a huge number
        ### ln(expA + expB) = ln(expA + exp(A+x)) = ln(expA (1+exp^x) ) = A + ln(1+exp^x)
        diff_in_delta = delta_0 - delta_1
        
        # 2. new EV
        #EV_new['0'] = transition['0'] @ np.log(np.exp(delta_0) + np.exp(delta_1))
        #EV_new['1'] = transition['1'] @ np.log(np.exp(delta_0) + np.exp(delta_1))
        # the above 2 lines do not work globally when we try theta, for the above inf reasons
        EV_new['0'] = transition['0'] @ (delta_1 + np.log(1 + np.exp(diff_in_delta))) + gamma
        EV_new['1'] = transition['1'] @ (delta_1 + np.log(1 + np.exp(diff_in_delta))) + gamma
        
        # 3. updata:
        diff = np.abs(EV_new['0'] - EV['0']).sum() + np.abs(EV_new['1'] - EV['1']).sum()
        EV = EV_new.copy()
        i = i + 1
    
    print('total iter : {}'.format(i))
    print('final diff : {}'.format(diff))

    return EV_new

# ---------------------------------------------------------------------------------
# estimate EV: H-M inversion
# ---------------------------------------------------------------------------------

def HM_inversion_LHS(transition, P, beta = 0.95):

    num_of_states = transition['0'].shape[0]
    
    # 1. P*transition
    combine_P_0 = (1-P[:, np.newaxis])*transition['0']
    combine_P_1 = P[:, np.newaxis]*transition['1']
    to_inv = np.eye(num_of_states) - beta*(combine_P_0 + combine_P_1)

    
    return np.linalg.inv(to_inv)
    #return to_inv

def HM_inversion_RHS(u, P):

    gamma = 0.577216
    # phi 
    precision = 10**-300
    P[P<precision] = precision
    P[P>1-precision] = 1-precision
    phi_0 = gamma - np.log(1-P)
    phi_1 = gamma - np.log(P)

    # calculation
    total_u_0 = u['0'] + phi_0
    total_u_1 = u['1'] + phi_1

    # total

    return (1-P) * total_u_0 + P * total_u_1









