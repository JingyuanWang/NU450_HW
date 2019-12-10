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
import itertools as it
import copy
import importlib


# random seed
np.random.seed(seed=13344)


# ---------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------

class Entry_MLE_Estimation:
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

    # I. basic functions -------------------------------------------------------------------
    def prepare_for_MLE(self):
        '''Preparation: save observables '''

        self.data_N_A = self.entry_sample.sample[0,:]
        self.data_N_B = self.entry_sample.sample[1,:]
        self.data_N_m = self.entry_sample.sample[2,:]
        self.data_X_m = self.entry_sample.sample[3,:]
        

        return

    def extract_true_value(self):
        '''Preparation: save true values '''
        
        self.true_theta = self.entry_sample.sample_theta_true
        self.true_prob  = self.entry_sample.sample_prob_entry

        return 


    # II. Objective and contraints ---------------------------------------------------------
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




