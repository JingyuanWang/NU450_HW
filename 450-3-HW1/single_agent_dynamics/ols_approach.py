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
while(os.path.basename(dirpath) != "NU450_HW") and (i<=10):
    dirpath = os.path.dirname(dirpath)
    i = i + 1
targetdir = dirpath + '/450-3-HW1'
if targetdir not in sys.path:
    sys.path.insert(0,targetdir)
i = 0
while(os.path.basename(dirpath) != "GitHub") and (i<=10):
    dirpath = os.path.dirname(dirpath)
    i = i + 1
targetdir = dirpath + '/tools'
if targetdir not in sys.path:
    sys.path.insert(0,targetdir)

from general import data_format
from single_agent_dynamics import est_func_step1_transitions
from single_agent_dynamics import est_func_step2_estimate_EV

importlib.reload(data_format)
importlib.reload(est_func_step1_transitions)
importlib.reload(est_func_step2_estimate_EV)

# ---------------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------------


def _correct_prob_2log(prob):
    
    precision = 10**-300
    prob[prob<precision] = precision
    
    return prob

# ---------------------------------------------------------------------------------
# class 1 
# ---------------------------------------------------------------------------------

class ols_scott2003:
    """This class take observations at state-var level for step2. The length of any obs should = length of state 
    The raw df is taken, in case we want to joint estimate step1 and step2. Otherwise, the transition matrix is not updating"""

    def __init__(self, observation_class, joint_est=False):
        """Here we observations at state-var level. The length of any obs should = length of state """
        
        # save state x, rc, rc_names, and prob
        self.state_level_df = observation_class.state_var_level_obs.copy()
        self.obs_df = observation_class.df.copy()

        # some info to calculate transition matrix
        if joint_est:
            self.rc_cutoffs = observation_class.rc_cutoffs.copy()
        else:
            self.transition = observation_class.transition_fromstep1.copy()
            self.rho = observation_class.rho_fromstep1 
            self.sigma_rho = observation_class.sigma_rho_fromstep1 
       
        self.joint_est = joint_est

        # 
        self.gamma = 0.577216


        

        return


    # ---- II prepare:  -------------------------------------------
    def _construct_reg_df(self):

        self.df2reg_statelevel = self._contruct_reg_df_state_level()
        self.df2reg_obs_level = self._contruct_reg_df_obs_level()

        return

    def _contruct_reg_df_state_level(self, beta = 0.95):    

        df = self.state_level_df.copy()
        P = self.state_level_df['P'].values.copy()
        num_rc_grid = len(df.rc.unique())
        num_x = len(df.x.unique())    

        # 1. Prob
        precision = 10**-300
        prob_0 = 1-P
        prob_1 = P.copy()
        #prob_0[prob_0<precision] = precision
        #prob_1[prob_1<precision] = precision
        
        # 2. dependent var, LHS:
        y_part_1 = np.log(prob_0) - np.log(prob_1) 
        y_part_2 = beta * (np.log(prob_1) - np.log(np.tile(prob_1[:num_rc_grid], num_x )) )
        
        # part 2 is the "next day state", so x should be x+1, except for 7
        y_part_2 = np.hstack([y_part_2[num_rc_grid:],  y_part_2[-num_rc_grid:] ] )
        
        # y is a combination of first part and second part. (not simple add, should rotation different part2 RCs for each part1)
        y_part_1 = np.repeat(y_part_1, num_rc_grid)
        y_part_2 = y_part_2.reshape( (num_x, num_rc_grid) ).repeat(num_rc_grid, axis = 0).reshape( (num_x * num_rc_grid * num_rc_grid, ) )
        y = y_part_1 + y_part_2

        # 3. indepedent
        # independent vars are at the same state as y_part_1
        x = df.x.values.copy()
        rc = df.rc.values.copy()
        x = np.repeat(x, num_rc_grid)
        rc = np.repeat(rc, num_rc_grid)
        
        # output
        df_to_reg = pd.DataFrame( [y, y_part_1, y_part_2, x, rc] ).T
        df_to_reg.rename(columns = {0:'y',1:'y_part1_deltadiff', 2:'y_part2_EUdff', 3:'x', 4:'rc'}, inplace = True)
        df_to_reg = df_to_reg.merge(df, on = ['x', 'rc'])

        return df_to_reg

    def _contruct_reg_df_obs_level(self, beta = 0.95):
        
        df = self.obs_df.copy()
        
        # 1. get next day state variable
        next_period = df[ ['i', 't','x', 'rc', 'grid_mean'] ].groupby('i').apply(lambda sub_df: sub_df.shift(-1))
        next_period.rename(columns = {'x':'x_next_real', 'rc':'rc_next', 'grid_mean':'grid_mean_next'}, inplace = True)
        next_period['t'] = next_period['t']-1
        df = df.merge(next_period, on = ['i', 't'])
        df['x_a1_next'] = 1
        df['x_a0_next'] = df['x'] + 1
        df.loc[ df['x_a1_next'] == 8, 'x_a1_next' ] = 7
        
        # 2. get probability of the next state: if at t, chose a=0
        to_merge = self.state_level_df[ ['rc', 'x', 'P'] ].copy()
        to_merge.rename(columns = {'P':'P1_a0_next', 'rc':'grid_mean', 'x':'x_a0_next'}, inplace = True)
        df = df.merge( to_merge , left_on = ['grid_mean', 'x_a0_next'], right_on =['grid_mean', 'x_a0_next'] )
        
        # 3. merge in next stage conterfactural probability: if at t, chose a=1
        to_merge = to_merge[to_merge['x_a0_next']==1]
        to_merge = to_merge.rename( columns = {'P1_a0_next':'P1_a1_next'}).drop(columns = ['x_a0_next'])
        df = df.merge( to_merge , left_on = ['grid_mean'], right_on =['grid_mean'] )
        
        # ---- finish constructing continuation df
        df = df.dropna()

        # 4. calculate y
        P = df.P.values.copy()
        #prob_0 = _correct_prob_2log(1-P)
        #prob_1 = _correct_prob_2log(P)
        prob_0 = 1-P
        prob_1 = P
        y_part1_u = np.log(prob_0) - np.log(prob_1)
        
        #P1_a0_next = _correct_prob_2log(df.P1_a0_next.values.copy())
        #P1_a1_next = _correct_prob_2log(df.P1_a1_next.values.copy())
        P1_a0_next = df.P1_a0_next.values.copy()
        P1_a1_next = df.P1_a1_next.values.copy()
        y_part2_EU = np.log(P1_a0_next) - np.log(P1_a1_next)
        
        df['y'] = y_part1_u + beta * y_part2_EU
        
        return df.sort_values(['i', 't']).reset_index(drop=True)





 


    # ---- III estimation  -----------------------------------------












