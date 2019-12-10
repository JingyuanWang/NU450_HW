'''
# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose:
# define class: Discrete Choice
#
# methods include:
# I. simulate consumer choices
# II. estimate coeff using BLP
# ------------------------------------------------------------------------
'''


import numpy as np
import pandas as pd
import os,sys,inspect
import scipy.stats as stats
import scipy.integrate as integrate
import econtools 
import econtools.metrics as mt
import itertools as it
import copy
import importlib


# random seed
np.random.seed(seed=13344)


# ---------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------

class Entry_Estimation:
    '''Estimateion: MLE+MPEC '''

    # get the main dataframe (and get the set of variables)
    def __init__(self, Entry_Model):
        '''initialize a dataclass for discrete demand system analysis
        
        Input:
        -- Entry_Model, classed defined in entry module in the same folder. 
                        This class includes a sample and several basic functions about the true value of the sample. '''
        
        self.entry_sample = Entry_Model 

        # parameters
        self.num_of_market = Entry_Model.sample_size

        # prepare for estimation
        self.prepare_for_MLE()
        self.extract_true_value()
        self.prepare_for_twostepOLS()

    # I. basic functions -------------------------------------------------------------------
    def prepare_for_MLE(self):
        '''Preparation: save observables '''

        self.data_N_A = self.entry_sample.sample[0,:]
        self.data_N_B = self.entry_sample.sample[1,:]
        self.data_N_m = self.entry_sample.sample[2,:]
        self.data_X_m = self.entry_sample.sample[3,:]
        
        return

    def prepare_for_twostepOLS(self):
        '''Preparation: save observables in dataframe '''
        
        self.sample_df = pd.DataFrame(data = self.entry_sample.sample.T, columns = ['N_A', 'N_B', 'N_m', 'X_m'])  

        self.extract_true_value()
        self.sample_df['P_0'] = self.true_prob[0,:]
        self.sample_df['P_A'] = self.true_prob[1,:]
        self.sample_df['P_B'] = self.true_prob[2,:]

        return

    def extract_true_value(self):
        '''Preparation: save true values '''
        
        self.true_theta = self.entry_sample.sample_theta_true
        self.true_prob  = self.entry_sample.sample_prob_entry

        return 


    # II. MLE ---------------------------------------------------------
    def obj_loglikelihood(self, theta_input):

        # data
        X_m = self.data_X_m 
        N_m = self.data_N_m 
        N_A = self.data_N_A
        N_B = self.data_N_B

        # Step1: the current estimated probability of entering at A, B
        prob_est = self.entry_sample.Solve_prob_of_entry_SymmetricREE(theta_input, X_m, N_m)

        prob_est_0 = prob_est[0,:]
        prob_est_A = prob_est[1,:]
        prob_est_B = prob_est[2,:]

        # Step2: likelihood of the sample
        loglikelihood = N_A * np.log(prob_est_A) + N_B * np.log(prob_est_B) + (N_m - N_A - N_B) * np.log(prob_est_0)
        obj = sum(loglikelihood)

        return obj

    # III. Two-step ------------------------------------------------------
    def OLS(self, true_prob=False):

        df = self._OLS_gen_var(true_prob)

        res_stage1 = self._OLS_first_stage(df)
        res_stage2 = self._OLS_second_stage(df)

        # extract parameters
        se = np.insert(res_stage1.se.values, [2,4], res_stage2.se.values)
        beta = np.insert(res_stage1.beta.values, [2,4], res_stage2.beta.values)
        p = np.insert(res_stage1.pt.values, [2,4], res_stage2.pt.values)
        res = pd.DataFrame(np.array([beta,se,p]).T, 
                     index = ['beta0', 'beta1', 'beta2', 'gamma1', 'gamma2', 'gamma2-gamma1'], 
                     columns = ['beta', 'se', 'p>t'])

        return res

    def _OLS_gen_var(self, true_prob=False):

        # 1. get data
        df = self.sample_df.copy()
        df = df.replace(0, 0.001)

        # 2. generate variables
        if true_prob:
            # dependent:
            df['log_PA_PB'] = np.log(df['P_A']/df['P_B'])  
            df['log_PB_P0'] = np.log(df['P_B']/df['P_0']) 

            # independent var:
            df['const'] = 1
            df['NA_NB_diff'] = (df['P_A'] - df['P_B'])*df['N_m']    

            df['N_A'] = df['P_A']*df['N_m']
            df['N_B'] = df['P_B']*df['N_m']

        else:
            # dependent:
            df['log_PA_PB'] = np.log(df['N_A']/df['N_B'])
            df['N_0'] = df['N_m'] - df['N_A'] - df['N_B']
            df['log_PB_P0'] = np.log(df['N_B']/df['N_0'])

            # independent var:
            df['const'] = 1
            df['NA_NB_diff'] = df['N_A'] - df['N_B']

        return df.replace([np.inf, -np.inf], np.nan).dropna()
        #return df

    def _OLS_first_stage(self, df):

        res = mt.reg(df,'log_PB_P0', ['const', 'X_m','N_B', 'N_A'] )

        return res

    def _OLS_second_stage(self, df):

        res = mt.reg(df,'log_PA_PB', ['const','NA_NB_diff'] )

        return res




