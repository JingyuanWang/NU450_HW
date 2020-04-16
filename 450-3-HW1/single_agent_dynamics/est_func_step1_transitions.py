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
# estimate rho 
# ---------------------------------------------------------------------------------


def est_rho(df_input):

    df = df_input.copy()
    df.sort_values( ['i','t'], inplace = True )
    df['rc_lag'] = (df[ ['i','rc', 't'] ].groupby('i')
                          .apply( lambda df: df.sort_values(by = 't').shift(1) )['rc']
                          .values)
    df.dropna(inplace =True)       

    # regress
    y = df['rc'].values[:, np.newaxis]
    X = np.vstack( [ np.ones(len(y)), df['rc_lag'].values ] ).T    
    

    rho = np.linalg.inv(X.T @ X) @ X.T @ y
    e   = y - X@rho
    sigma_rho =  e.T@e / (len(e)-1)
    sigma_rho = np.sqrt(sigma_rho.flatten()[0])

    return rho, sigma_rho

# ---------------------------------------------------------------------------------
# estimate transition matrix
# ---------------------------------------------------------------------------------

def transition_mat(rho, sigma_rho, grid_cutoffs, show_transition_mat = False):
    '''return a diction of 2 transition mat: a=1 and a=0 '''
    
    rc_transition = rc_transition_mat(rho, sigma_rho, grid_cutoffs, show_transition_mat)
    transition = _expand_rc_transition_to_allstatevar_transition(rc_transition, show_transition_mat)
    
    return transition 


def rc_transition_mat(rho, sigma_rho, grid_cutoffs, show_transition_mat = False):
    
    # calculate eps cutoffs: lb - (rho0 + rho1*p_{t-1}) < eps < ib - (rho0 + rho1*p_{t-1})
    p = np.arange(grid_cutoffs[0], grid_cutoffs[-1], (grid_cutoffs[-1] -grid_cutoffs[0]) / 100)
    epsilon_cutoff = np.ones( (100, len(grid_cutoffs)) ) * grid_cutoffs[np.newaxis, : ] - (rho[0] + rho[1] * p)[:, np.newaxis]
    
    # cdf of eps
    cdfs = stats.norm.cdf(epsilon_cutoff, loc=0, scale=sigma_rho)
    if cdfs[:,-1].min() < 1-1e-3 :
        print('ERROR !!!!! maximum possible value exceed the lower bound, min(cdf_ub) = {}'.format(cdfs[:,-1].min()))
    if cdfs[:,0].max() > 1e-3:
        print('ERROR !!!!! minimum possible value exceed the upper bound, max(cdf_lp) = {}'.format(cdfs[:,0].max()) )
    
    # get transition matrix for each point in p (there are 100 points in total)
    cdfs[:,0] = 0
    cdfs[:,-1] = 1
    transition =  cdfs[:,1:] - cdfs[:,:-1]

    # aggregate 100 points in p into those bins defined by the grid_cutoffs
    n_bins = len(grid_cutoffs) -1
    grid_transition = np.zeros( (n_bins, n_bins) )
    for i in range(n_bins):
        grid_transition[i,:] = np.mean(transition[ (p >= grid_cutoffs[i]) & (p<= grid_cutoffs[i+1]), : ], axis = 0)
    
    if show_transition_mat:
        fig = plt.figure(figsize = (10,3))
        plt.subplot(1, 2, 1)
        sns.heatmap(transition, cmap = 'YlGnBu')
        plt.title('(1) before integral into grid')
        plt.subplot(1, 2, 2)
        sns.heatmap(grid_transition, cmap = 'YlGnBu')
        plt.title('(2) grid transition ')
        plt.show()

    return grid_transition



def _expand_rc_transition_to_allstatevar_transition(rc_transition, show_transition_mat=False):

    # 1. re set the state variable to x and rc, and x have 7 possible values
    n_rc_grids = rc_transition.shape[0]
    transition_expand = np.kron(np.eye( 7 ), rc_transition )
    # (1) if a = 0
    transition_0 = transition_expand.copy()
    # transition of x:
    transition_0[: n_rc_grids*6, n_rc_grids:] = transition_expand[: n_rc_grids*6, : n_rc_grids*6]
    transition_0[: n_rc_grids*6, :n_rc_grids] = 0
    # Final: the first n rows represent the probability from x=1, rc from the n grids to each (7*n) state. The first n columns are to x=1 states.
    
    # （2） transition if a = 1
    transition_1 = np.zeros( (n_rc_grids*7, n_rc_grids*7) ) 
    transition_1[:, :n_rc_grids] = np.kron(np.ones( (7,1) ), rc_transition )

 
    # 2. save 
    transition = {'0':transition_0, '1':transition_1}
    if show_transition_mat:
        print('# ---- transition matrix for all states ---------- ')
        fig = plt.figure(figsize = (10,3))
        plt.subplot(1, 2, 1)
        sns.heatmap(transition['0'], cmap="YlGnBu")
        plt.subplot(1, 2, 2)
        sns.heatmap(transition['1'], cmap="YlGnBu")
        plt.show()

    return transition

# ---------------------------------------------------------------------------------
# estimate transition matrix
# ---------------------------------------------------------------------------------









