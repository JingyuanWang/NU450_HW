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


from single_agent_dynamics import est_func_step1_transitions
from single_agent_dynamics import est_func_step2_estimate_EV


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
    



# ---------------------------------------------------------------------------------
# class 1 
# ---------------------------------------------------------------------------------

class observations:

    def __init__(self, df_input, n_rc_grids = 6, weight = True, true_parameters = None):


        # basic adjustment: grid state var & estimate probabilities
        print('# ------------------------------------------------------------------ ')
        print('# DATA CLEANING: cut state var into grid and show the data pattern  ')
        print('# ------------------------------------------------------------------ ')

        self.df, self.rc_cutoffs = get_grid(df_input, 'rc', n_bins = n_rc_grids, weight = weight)
        self._calculate_action_prob_bystate()
        self._plot_initial_transition_mat()
        self._get_state_var_id()

        # prepae for est
        self.obs_to_state_var_level()

        return
    
    # ---- I clean data -------------------------------------------
    def _calculate_action_prob_bystate(self):

        df = self.df.copy()
        prob = df[ ['x','rc_grid','a'] ].groupby(['x','rc_grid' ]).agg( np.mean ).rename(columns = {'a':'P'})
        prob_mat = prob.pivot_table(values = 'P', index = 'rc_grid', columns = 'x')
        print('# ---- average probability in each bin ---------- ')
        fig, ax = plt.subplots()
        sns.heatmap(prob_mat, cmap="YlGnBu")
        plt.show()

        self.action_prob_bystate = prob_mat
        self.df = self.df.merge(prob, on = ['x','rc_grid']).sort_values(['i','t']).reset_index(drop =True)

        return

    def _plot_initial_transition_mat(self):
        
        rho, sigma_rho = est_func_step1_transitions.est_rho(self.df)
        self.transition_fromstep1 = est_func_step1_transitions.transition_mat(rho, sigma_rho, self.rc_cutoffs, show_transition_mat = True)
        self.rho_fromstep1 = rho
        self.sigma_rho_fromstep1 = sigma_rho

        return

    def _get_state_var_id(self):

        df = self.df.copy()
        state_var_dict = df[['x','rc_grid_id']].drop_duplicates()
        
        state_var_dict = (state_var_dict.sort_values( ['x', 'rc_grid_id'] )
                               .reset_index(drop=True).reset_index()
                               .rename(columns = {'index':'state_var_id'}) )

        self.df = self.df.merge(state_var_dict, on = ['x', 'rc_grid_id']).sort_values(['i','t']).reset_index(drop =True)
        self.state_var_dict = state_var_dict

        return 

    # ---- II prepare for estimation: collapse to state-var level  -------------------------------------------
    def obs_to_state_var_level(self):

        # 1. save: x, mean of rc (weighted),  prob of action and rc
        df = self.df.copy()
        df1 = df[['state_var_id','x' ,'rc_grid_id','rc_grid', 'grid_mean', 'P']].drop_duplicates().sort_values('state_var_id').reset_index(drop =True)
        df1.rename(columns = {'grid_mean':'rc'}, inplace = True)

        # 2. number of obs in each state
        df2 = df[['state_var_id', 'x']].groupby('state_var_id').agg('count').rename(columns = {'x':'num_of_obs'})

        # 3. save
        output = df1.merge(df2, on='state_var_id')
        self.state_var_level_obs = output

        return



    


# ---------------------------------------------------------------------------------
# class 2
# ---------------------------------------------------------------------------------

class estimation:
    """This class take observations at state-var level for step2. The length of any obs should = length of state 
    The raw df is taken, in case we want to joint estimate step1 and step2. Otherwise, the transition matrix is not updating"""

    def __init__(self, observation_class, joint_est=False):
        """Here we observations at state-var level. The length of any obs should = length of state """
        
        # save state x, rc, rc_names, and prob
        self.df = observation_class.state_var_level_obs.copy()

        # some info to calculate transition matrix
        if joint_est:
            self.rc_cutoffs = observation_class.rc_cutoffs.copy()
            self.df_obs_to_estimate_rho = observation_class.df.copy()
        else:
            self.transition = observation_class.transition_fromstep1.copy()
            self.rho = observation_class.rho_fromstep1 
            self.sigma_rho = observation_class.sigma_rho_fromstep1 
       
        self.joint_est = joint_est

        # to array
        self._preparation_save2array()

        return


    # ---- II prepare:  -------------------------------------------

    def _preparation_save2array(self):

        self.x = self.df['x'].values.copy()
        self.rc = self.df['rc'].values.copy()
        self.action_prob = self.df['P'].values.copy()
        self.num_of_chosing0 = self.df['num_of_obs'].values * (1-self.df['P'].values)
        self.num_of_chosing1 = self.df['num_of_obs'].values * self.df['P'].values

        # prepare for H-M inversion
        transition = self.transition.copy()
        P = self.df.P.values.copy()
        self.HM_inv_LHS = est_func_step2_estimate_EV.HM_inversion_LHS(transition, P)

        return 

    def _combine_parameters(self, theta):

        rho, sigma_rho = est_func_step1_transitions.est_rho(self.df)
        par_step1 = np.append(rho.flatten(), sigma_rho)
        parameters = np.append( theta , par_step1 )

        return parameters

    def _separate_parameters(self, parameters):
        theta = parameters[0:2]
        rho = parameters[2:4]
        sigma_rho = parameters[4]

        return theta, rho, sigma_rho

    # ---- III given parameter, estimate likelihood -------------------------------------------

    def get_loglikelihood(self, parameters, method = 'fixed_point', print_details = False):

        if self.joint_est:
            if len(parameters) != 5:
                print('Error: length of parameters != 5')
            else:
                loglikelihood = self.get_loglikelihood_joint_est(parameters)

        else:
            if len(parameters) != 2:
                print('Error: length of parameters != length of theta = 2')
            else:
                theta = parameters
                loglikelihood = self.get_loglikelihood_step2_est(theta, beta = 0.95, method = method, print_details = print_details)

        return loglikelihood

    def get_loglikelihood_step2_est(self, theta, method = 'fixed_point', beta=0.95, print_details = False):
        '''method: fixed_point or H-M '''

        # 1. get values of each choice
        EV, u, transition = self._get_continuation_values(theta, method, beta, print_details = print_details)

        # calculate payoff of each choice
        delta_0 = u['0'] + beta*EV['0'] 
        delta_1 = u['1'] + beta*EV['1'] 

        # 4. calculate the likelihood
        # to avoid exp(something large) = inf:
        #     for each state, 
        #         devide the denominator and numerator by exp_delta_0
        prob_0 = 1/(np.exp(delta_1-delta_0) + 1)
        prob_1 = 1-prob_0
        # this is likelihood by state

        # 5. loglikehood:
        precision = 10**-300 # np.log() = -inf for smaller numbers
        # take log for each state 
        prob_0[prob_0<precision] = precision
        prob_1[prob_1<precision] = precision
        log_p_0 = np.log(prob_0)
        log_p_1 = np.log(prob_1)

        loglikelihood = log_p_0 * self.num_of_chosing0 + log_p_1 * self.num_of_chosing1
        loglikelihood = loglikelihood.sum()
        
        return -loglikelihood
    

    def get_loglikelihood_joint_est(self, parameters):
        
        # ...... 
        theta, rho, sigma_rho = self._separate_parameters(parameters)

        self._get_transition_mat(parameters)
        # ..... 

        loglikelihood = 9999

        return -loglikelihood


    # --------------------
    # prepatation functions
    # --------------------
    def _get_perperiod_u(self, theta):

        u_0 = -self.x * theta[0]
        u_1 = -self.rc * theta[1] 

        u = {}
        u['0'] = u_0
        u['1'] = u_1

        return u

    def _get_continuation_values(self, theta, method = 'fixed_point' , beta = 0.95, print_details = False):

        # 1. get perperiod payoff 
        u = self._get_perperiod_u(theta)

        # 2. get transition matrix (for free)
        transition = self.transition.copy() # note this is a dictionary

        # 3. get continuation value
        if method == 'fixed_point':
            EV = est_func_step2_estimate_EV.find_fixed_point(u, transition, beta =beta, print_details = print_details)
        elif method == 'H-Minversion':
            P = self.df.P.values.copy()
            EV = self.EV_from_HMinversion(u, transition, P)
        else:
            print('please chose method between fixed_point and H-Minversion')
            return 

        return EV, u, transition


    def _get_transition_mat(self, parameters_input, show_transition_mat=False):
        
        parameters = parameters_input.copy()
        # 1. p_transaction matrix
        rho = parameters[1:3]
        sigma_rho = parameters[3]
        transition = est_func_step1_transitions.transition_mat(rho, sigma_rho, self.rc_cutoffs, show_transition_mat = show_transition_mat)

        return transition



    # --------------------
    # H-M inversion functions
    # --------------------
    def EV_from_HMinversion(self, u, transition, P):

        # calculate the continuation value of each state (realization)
        inv_LHS = self.HM_inv_LHS.copy()
        RHS = est_func_step2_estimate_EV.HM_inversion_RHS(u, P)
        
        V = inv_LHS @ RHS

        # calculate the Expected continuation value , given current state
        EV = {}
        EV['0'] = transition['0'] @ V
        EV['1'] = transition['1'] @ V

        return EV














