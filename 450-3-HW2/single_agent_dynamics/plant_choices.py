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

def get_grid(df_input, varname, n_bins = 6, weight = True):

    # 1. cut
    df = df_input.copy()

    buffer_value = 0.1
    cutoffs = np.linspace(df[varname].min()-buffer_value  , df[varname].max()+buffer_value , n_bins+1)
    (cat_var , cutoffs ) = pd.cut(df[varname], bins = cutoffs, retbins = True)


    print('# ---- number of obs in each bin ---------- ')
    print( cat_var.value_counts().sort_index() )
    df['{}_grid'.format(varname)] = cat_var

    # 2 group id
    grid_dict = (cat_var.value_counts()
                .sort_index().reset_index().reset_index()
                .rename(columns = {'level_0':'{}_grid_id'.format(varname), 
                    'index':'{}_grid'.format(varname)}))[['{}_grid_id'.format(varname),'{}_grid'.format(varname)]]
    df = df.merge(grid_dict, on = ['{}_grid'.format(varname)]) 

    # 3.group mean
    unweighted_mean = (cutoffs[:-1] + cutoffs[1:])/2
    unweighted_mean = pd.DataFrame( [unweighted_mean] ).T.reset_index().rename(columns = {'index':'{}_grid_id'.format(varname),
        0:'grid_mean'})
    weighted_mean = df[ [varname, '{}_grid'.format(varname)]  ].groupby('{}_grid'.format(varname)).agg(np.mean).rename(columns = {varname:'grid_mean'})
    if weight:
        df = df.merge(weighted_mean, on = '{}_grid'.format(varname))
#        return df, weighted_mean
    else:
        df = df.merge(unweighted_mean, on = '{}_grid_id'.format(varname))
#        return df, unweighted_mean

    return df.sort_values( ['i','t'] ).reset_index(drop=True), cutoffs
    

def _basic_clean(df_input):

    df = df_input.copy()
    df = df[ ['frsnumber', 'year', 'quarter', 'quarter_str', 'time_id',
              'inspection', 'fine', 'violation', 'hpv_recorded',
              'investment', 
              'ordered_violator', 'compliance', 'violator_nothpv', 'hpv_status', 'lag_investment', 
              'lag_ordered_violator', 'lag_compliance', 'lag_violator_nothpv', 'lag_hpv_status', 'lag2_investment']  ]

    return df 

# ---------------------------------------------------------------------------------
# class 1 
# ---------------------------------------------------------------------------------

class observations:

    def __init__(self, df_input):


        # basic adjustment: grid state var & estimate probabilities
        print('# ------------------------------------------------------------------ ')
        print('# DATA CLEANING: basic clean and show the data pattern  ')
        print('# ------------------------------------------------------------------ ')

        #self.df, self.rc_cutoffs = get_grid(df_input, 'rc', n_bins = n_rc_grids, weight = weight)
        # keep the above row for DAV. details see HW1

        self.df = _basic_clean(df_input)
        self._calculate_action_prob_bystate( state_var = ['ordered_violator','lag_investment'], choice_var = ['investment'])
        self._plot_transition_mat()
        self._get_state_var_id()


        return
    
    # ---- I clean data -------------------------------------------
    def _calculate_action_prob_bystate(self, state_var = [], choice_var = []):

        df = self.df.copy()
        prob = df[ state_var + choice_var ].groupby(state_var).agg( np.mean ).rename(columns = {choice_var[0]:'P'})
        prob_mat = prob.pivot_table(values = 'P', index = state_var[0], columns = state_var[1])
        print('# ---- investment CCP in each state ---------- ')
        fig, ax = plt.subplots()
        sns.heatmap(prob_mat, cmap="YlGnBu", annot=True)
        plt.show()

        self.action_prob_bystate_wide = prob_mat
        self.action_prob_bystate_long = prob 
        self.df = self.df.merge(prob, on = state_var ).sort_values(['frsnumber','time_id']).reset_index(drop =True)

        return

    def _plot_transition_mat(self):
        
        df = self.df.copy()
        prob = (df[ ['lag_ordered_violator', 'lag2_investment', 
                    'lag_investment',
                    'compliance', 'violator_nothpv', 'hpv_status' ]  ]
                    .groupby(['lag_ordered_violator', 'lag2_investment', 'lag_investment' ] )
                    .agg( np.mean ) ).reset_index()

        prob0 = prob[prob['lag_investment']==0].sort_values( ['lag2_investment', 'lag_ordered_violator' ] ).reset_index(drop=True)
        prob1 = prob[prob['lag_investment']==1].sort_values( ['lag2_investment', 'lag_ordered_violator' ]).reset_index(drop=True)          

        # 1. generate prob0 matrix
        prob0.rename(columns = {'compliance':'compliance0',
                                'violator_nothpv':'violator_nothpv0',
                                'hpv_status':'hpv_status0'} , inplace = True)
        prob0[ 'compliance1' ] = 0
        prob0[ 'violator_nothpv1' ] = 0
        prob0[ 'hpv_status1' ] = 0        

        # 2. generate prob1 matrix       
        prob1 = (prob1.sort_values( ['lag2_investment', 'lag_ordered_violator' ])
                 .reset_index(drop=True)
                 .rename(columns = {'compliance':'compliance1',
                                'violator_nothpv':'violator_nothpv1',
                                'hpv_status':'hpv_status1'} )
                )
        prob1[ 'compliance0' ] = 0
        prob1[ 'violator_nothpv0' ] = 0
        prob1[ 'hpv_status0' ] = 0        

        prob1 = prob1[list(prob1.columns[:3]) + list(prob1.columns[-3:]) + list(prob1.columns[3:-3])]

        # 3. save probability of not investing, compliance, laginv, as the probability of investing in these state is not defined
        compliance_laginv0 = prob0.loc[0].copy()
        compliance_laginv1 = prob0.loc[3].copy()
        prob1.loc[4] = compliance_laginv0
        prob1.loc[5] = compliance_laginv1


        prob1['lag_investment'] = 1 
        prob1 = prob1.sort_values(['lag2_investment', 'lag_ordered_violator']).reset_index(drop=True)  

        #prob1.iloc[0,6:9] = prob1.iloc[0,3:6].values
        #prob1.iloc[3,6:9] = prob1.iloc[3,3:6].values

        
        # 4. plot
        print('# ---- transition probability | not invest ---------- ')
        fig, ax = plt.subplots()
        sns.heatmap(prob0[ prob0.columns[3:] ].T, cmap="YlGnBu", annot=True)
        plt.show()
        print('# ---- transition probability | invest ---------- ')
        fig, ax = plt.subplots()
        sns.heatmap(prob1[ prob1.columns[3:] ].T, cmap="YlGnBu", annot=True)
        plt.show()

        print('# ---- for output, transition mat ---------- ')
        fig = plt.figure(figsize = (15,3))
        plt.subplot(1, 2, 1)
        sns.heatmap(prob0[ prob0.columns[3:] ].T, cmap="YlGnBu", annot=True)
        plt.title('transition probability | not invest')
        plt.subplot(1, 2, 2)
        sns.heatmap(prob1[ prob1.columns[3:] ].T, cmap="YlGnBu", annot=True)
        plt.title('transition probability | invest ')
        plt.show()


        # 5. save 
        self.transition_mat0 = prob0
        self.transition_mat1 = prob1
        self.transition = { '0':prob0[prob0.columns[3:]].values.T, '1':prob1[prob1.columns[3:]].values.T }

        return

    def _get_state_var_id(self):

        df = self.df.copy()
        state_var_dict = df[['lag_ordered_violator','lag2_investment']].drop_duplicates()
        
        state_var_dict = (state_var_dict.sort_values( ['lag2_investment', 'lag_ordered_violator'] )
                               .reset_index(drop=True).reset_index()
                               .rename(columns = {'index':'state_var_id'}) )

        self.df = self.df.merge(state_var_dict, on = ['lag2_investment', 'lag_ordered_violator']).sort_values(['frsnumber','time_id']).reset_index(drop =True)
        self.state_var_dict = state_var_dict

        return 

    # ---- II prepare for estimation: collapse to state-var level  -------------------------------------------




    




