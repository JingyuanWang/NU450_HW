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

importlib.reload(data_format)


# ---------------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------
# class 1 
# ---------------------------------------------------------------------------------

class ols_modified:
    """This class take observations at state-var level for step2. The length of any obs should = length of state 
    The raw df is taken, in case we want to joint estimate step1 and step2. Otherwise, the transition matrix is not updating"""

    def __init__(self, observation_class):
        """Here we observations at state-var level. The length of any obs should = length of state """
        
        # save state x, rc, rc_names, and prob
        self.obs_df = observation_class.df.copy()

        # transition
        self.transition = observation_class.transition.copy()
        self.transition_mat0  = observation_class.transition_mat0.copy()
        self.transition_mat1  = observation_class.transition_mat1.copy()
        self.state_var_dict = observation_class.state_var_dict.copy()
        
        # action probability
        CCP = observation_class.action_prob_bystate_long.reset_index()
        CCP['state_var_id'] = CCP['lag_investment'] * 3 + CCP['ordered_violator']
        CCP['lnP1'] = np.log(CCP['P'])
        CCP['lnP0'] = np.log(1-CCP['P'])
        CCP.loc[  CCP['ordered_violator'] == 0, 'lnP1' ] = 0 
        CCP.loc[  CCP['ordered_violator'] == 0, 'lnP0' ] = 0 

        self.CCP = CCP.sort_values('state_var_id').reset_index(drop = True)

        # 
        self.gamma = 0.577216

        # prepare
        self._prepare_for_est()
        self._merge_in_CCP()
        self._merge_in_transitionprob()


        return


    # ---- II prepare: contruct actual and counterfactual path -------------------------------------------
    def _prepare_for_est(self):
        ''' save the state '''
        
        # 1. actural time t realization
        df = self.obs_df.copy()
        # for state in which compliance == 1, we do not have choice probability or value of choice (LHS). Drop these rows
        df = df[ df['lag_compliance'] == 0 ]
        df = df[ ['frsnumber','time_id', 'state_var_id', 'lag2_investment', 'lag_ordered_violator' , 'lag_investment', 'ordered_violator'] ]

        # merge in state id 
        to_merge = self.state_var_dict.copy()
        to_merge.rename(columns = {'lag_ordered_violator': 'ordered_violator', 
                           'lag2_investment':'lag_investment',
                           'state_var_id':'state_var_id_actual'}, inplace = True)
        df = df.merge(to_merge ,on = ['lag_investment', 'ordered_violator'])
        df = df.sort_values( ['frsnumber','time_id'] ).reset_index(drop = True)
        
        df.rename(columns = {'lag_investment':'lag_investment_actual', 
                             'ordered_violator': 'ordered_violator_actual'} , inplace = True)


        # 2. counterfactual time t realization
        df['lag_investment_counterfactual'] = 1-df['lag_investment_actual']
        counterfactual_state_list = self._simulate_counterfactural(df)
        df['state_var_id_counterfactual'] = counterfactual_state_list

        # 3. state id 
        self.df_est = df.sort_values( ['frsnumber','time_id'] ).reset_index(drop = True)

        return

    def _simulate_counterfactural(self, df_est):

        state_list = []
        for i, inv, state in zip( range(len(df_est)),
                                  df_est['lag_investment_counterfactual'].tolist(), 
                                  df_est['state_var_id'].tolist() ):
            new_state = self._simulate(inv = inv, old_state = state)
            state_list.append(new_state)
            #if np.floor(i/50) == i/50:
            #    print(i)

        return state_list 

    def _simulate(self, inv, old_state):

        T0 = self.transition['0'].copy()
        T1 = self.transition['1'].copy()

        if inv == 0:
            new_state = np.random.choice(np.arange(0, 6), p= T0[:,old_state], size = 1 )
        if inv == 1:
            new_state = np.random.choice(np.arange(0, 6), p= T1[:,old_state], size = 1 )

        return float(new_state)

    
    # ---- III merge in all variables  -------------------------------------------
    def _merge_in_CCP(self):

        df = self.df_est.copy()

        # t-1 choice value 
        to_merge = self.CCP[['state_var_id', 'lnP0', 'lnP1' ] ]
        df = df.merge(to_merge, on = 'state_var_id')

        # time t actual path 
        to_merge = self.CCP[['state_var_id', 'lnP0', 'lnP1' ] ].rename(columns = {'lnP0':'lnP0_actual', 'lnP1': 'lnP1_actual',     
                                                                                     'state_var_id':'state_var_id_actual' })
        df = df.merge(to_merge, on = 'state_var_id_actual')

        # time t counterfactural 
        to_merge = self.CCP[['state_var_id', 'lnP0', 'lnP1' ] ].rename(columns = {'lnP0':'lnP0_counterfactual', 'lnP1': 'lnP1_counterfactual',    
                                                                                      'state_var_id':'state_var_id_counterfactual' })
        df = df.merge(to_merge, on = 'state_var_id_counterfactual').sort_values( ['frsnumber', 'time_id'] ).reset_index(drop = True)

        self.df_est = df 

        return 

    def _merge_in_transitionprob(self):

        df = self.df_est.copy()

        # time t actual path 
        to_merge0 = self.transition_mat0[ ['compliance0', 'violator_nothpv0', 'hpv_status0', 'compliance1', 'violator_nothpv1', 'hpv_status1'] ]
        to_merge1 = self.transition_mat1[ ['compliance0', 'violator_nothpv0', 'hpv_status0', 'compliance1', 'violator_nothpv1', 'hpv_status1'] ]
        to_merge0.columns = [ '{}_0_a'.format(x) for x in to_merge0.columns]
        to_merge1.columns = [ '{}_1_a'.format(x) for x in to_merge1.columns]
        df = df.merge(to_merge0, left_on = 'state_var_id_actual', right_index = True)
        df = df.merge(to_merge1, left_on = 'state_var_id_actual', right_index = True)

        # time t counterfactual path 
        to_merge0 = self.transition_mat0[ ['compliance0', 'violator_nothpv0', 'hpv_status0', 'compliance1', 'violator_nothpv1', 'hpv_status1'] ]
        to_merge1 = self.transition_mat1[ ['compliance0', 'violator_nothpv0', 'hpv_status0', 'compliance1', 'violator_nothpv1', 'hpv_status1'] ]
        to_merge0.columns = [ '{}_0_cf'.format(x) for x in to_merge0.columns]
        to_merge1.columns = [ '{}_1_cf'.format(x) for x in to_merge1.columns]
        df = df.merge(to_merge0, left_on = 'state_var_id_counterfactual', right_index = True)
        df = df.merge(to_merge1, left_on = 'state_var_id_counterfactual', right_index = True)


        self.df_est = df.sort_values( ['frsnumber', 'time_id'] ).reset_index(drop = True) 

        return

    