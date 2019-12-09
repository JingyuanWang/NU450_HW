'''
# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose:
# define class: entry 
#
# methods include:
# I. simulate consumer choices
# II. estimate coeff using MLE
# ------------------------------------------------------------------------
'''


import numpy as np
import pandas as pd
import os,sys,inspect
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
import itertools as it
import copy
import importlib


# random seed
np.random.seed(seed=13344)


# ---------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------
class Entry_Model:
    '''Samples for Discrete Choice Demand system estimation
    main part: 
    -- 1 dataframs: products
    -- 2 functions: 
       -- 
    -- Notation 
       parameters:
       -- J: number of products
       -- M: mumber of markets
       -- k: number of independent variables
       -- q: number of demand side moment conditions = number of IVs
       -- 2: number of supply side moment conditions (didn't assign a letter)

       variables:
       -- X:
       -- Z:
       -- W:
       -- supplysideX:
       -- supplysideZ:

       '''

    # get the main dataframe (and get the set of variables)
    def __init__(self):
        '''initialize a dataclass for discrete demand system analysis
        
        Input:
        -- DiscreteChoice, classed defined in DiscreteChoice module in the same folder '''
        


    # I. basic functions ---------------------------------------------------------------------

    def _extract_parameters(self,theta_input):
        '''parameter of interests: theta, 5-by-1 

        Input:
        -- theta: 5-by-1. array or list

        Output:
        -- 5 scalars '''

        theta = _check_and_covert_to_array(theta_input)

        beta_0 = theta[0]
        beta_1 = theta[1]
        beta_2 = theta[2]
        gamma_1 = theta[3]
        gamma_2 = theta[4]

        return (beta_0, beta_1, beta_2,gamma_1, gamma_2)


    # II. Solve for equilibirum --------------------------------------------------------------
    def expected_prob_To_result_prob(self, theta_input, X_m_input, N_m_input, expected_prob_A_input, expected_prob_B_input ):
        '''A function calculate market share using given (product social value) delta, using simulated integral 
        
        Input:
        --theta: 5 element list or array, parameters of interest
        --X_m: scalar or array, samle length as N_m, market level profit shifter
        --N_m: scalar or array, samle length as X_m, total number of firms.
        --expected_prob_A: scalar or array, same length as X_m and N_m. probability of entering A
        --expected_prob_B: scalar or array, same length as X_m and N_m. probability of entering B

        Output:
        --result_prob_0: scalar or array, same length as X_m and N_m. probability of not entering
        --result_prob_A: scalar or array, same length as X_m and N_m. probability of entering A
        --result_prob_B: scalar or array, same length as X_m and N_m. probability of entering B
        '''
        
        # 1. get parameters
        (beta_0, beta_1, beta_2,gamma_1, gamma_2) = self._extract_parameters(theta_input)
        if isinstance(X_m_input, int) or isinstance(X_m_input, float):
            X_m = X_m_input
            N_m = N_m_input
            expected_prob_A = expected_prob_A_input
            expected_prob_B = expected_prob_B_input
        else:
            X_m = _check_and_covert_to_array(X_m_input)
            N_m = _check_and_covert_to_array(N_m_input)
            expected_prob_A = _check_and_covert_to_array(expected_prob_A_input)
            expected_prob_B = _check_and_covert_to_array(expected_prob_B_input)


        # 2. calculate the score of each location
        score_0 = 0
        score_A = beta_0 + beta_1 * X_m + beta_2 - gamma_1 * expected_prob_A * N_m - gamma_2 * expected_prob_B * N_m
        score_B = beta_0 + beta_1 * X_m - gamma_1 * expected_prob_B * N_m - gamma_2 * expected_prob_A * N_m

        # 3. logit probability
        result_prob_0 = np.exp(score_0)/(np.exp(score_0)+np.exp(score_A)+np.exp(score_B))
        result_prob_A = np.exp(score_A)/(np.exp(score_0)+np.exp(score_A)+np.exp(score_B))
        result_prob_B = np.exp(score_B)/(np.exp(score_0)+np.exp(score_A)+np.exp(score_B))

        if isinstance(X_m_input, int) or isinstance(X_m_input, float):
            return np.array(([result_prob_0], [result_prob_A], [result_prob_B]))
        else:
            return np.array((result_prob_0, result_prob_A, result_prob_B))

    def Solve_prob_of_entry_SymmetricREE(self, theta_input, X_m_input, N_m_input):
        '''Solve for the fixed point of function self.expected_prob_To_result_prob(theta, X_m, N_m, ...) 

        Input:
        -- theta_input: 5 element array or list, parameters of interest
        -- X_m: scalar or vector, market level profit shifterr
        -- N_m: scalar or vector, number of firms in the market  

        Output:
        --prob: equilibrium probability of entering A or B
                size: for one market, (2, ) array; for multiple markets, (2, NumOfMarkets) array '''

        # 1. Solve for the fixed point for P_A and P_B
        # (1) scalar
        if isinstance(X_m_input, int) or isinstance(X_m_input, float):
            # initial value
            x0 = np.array([.5, .5])
            # solve
            fixed_point = opt.fixed_point(lambda x: self.expected_prob_To_result_prob(theta_input, X_m_input, N_m_input, x[0], x[1])[1:].flatten(), x0)

        # (2) vector
        else:
            # initial value: 2-by-NumberOfMarkets 
            x0 = np.ones( (2,len(X_m_input)) ) * 0.5
            # solve
            fixed_point = opt.fixed_point(lambda x: self.expected_prob_To_result_prob(theta_input, X_m_input, N_m_input, x[0,:], x[1,:])[1:,:], x0)

        # P_0 = 1-P_A - P_B
        prob_0 = 1-np.sum(fixed_point, axis = 0)
        prob   = np.append([prob_0],fixed_point,axis = 0)

        return prob

    def Plot_prob_of_entry(self, theta_input, X_m_input, figpath = None, figname = None, save = False):
        '''Plot probability of entering, given parameters and market characteristics
        Input:
        -- theta_input: 5 element array or list, parameters of interest
        -- X_m: scalar or vector, market level profit shifterr
        -- N_m: scalar or vector, number of firms in the market  

        Output:
        --fig'''
        
        # 1. solve for equilibrium probability of entering
        n = len(X_m_input)
        N_m = np.ones(n)*5
        prob_5 = self.Solve_prob_of_entry_SymmetricREE(theta_input, X_m_input, N_m)

        N_m = np.ones(n)*10
        prob_10 = self.Solve_prob_of_entry_SymmetricREE(theta_input, X_m_input, N_m)

        N_m = np.ones(n)*15
        prob_15 = self.Solve_prob_of_entry_SymmetricREE(theta_input, X_m_input, N_m)


        # 2. plot  
        fig = plt.figure(figsize=(16,6))

        plt.subplot(1, 2, 1)
        plt.plot(X_m_input, prob_5[1,:], alpha=0.5, color = 'maroon', label = 'N_m = 5')
        plt.plot(X_m_input, prob_10[1,:], alpha=0.5, color = 'red', label = 'N_m = 10')
        plt.plot(X_m_input, prob_15[1,:], alpha=0.5, color = 'orange', label = 'N_m = 15')

        plt.title('probability of entering at A') 
        plt.legend(loc = 'best')     
        plt.xlabel('Market characteristics X_m')
        plt.ylabel('Prob[entry]')      

        # subplot 2 PDF
        plt.subplot(1, 2, 2)
        plt.plot(X_m_input, prob_5[2,:], alpha=0.5, color = 'maroon', label = 'N_m = 5')
        plt.plot(X_m_input, prob_10[2,:], alpha=0.5, color = 'red', label = 'N_m = 10')
        plt.plot(X_m_input, prob_15[2,:], alpha=0.5, color = 'orange', label = 'N_m = 15')

        plt.title('probability of entering at B') 
        plt.legend(loc = 'best')
        plt.xlabel('Market characteristics X_m')
        plt.ylabel('Prob[entry]')

        # save
        if save:
            filename = figpath + '/' + figname + '.png'
            plt.savefig(filename, bbox_inches = 'tight')

        plt.close()

        return fig

# ---------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------


def _check_and_covert_to_array(input_variable):
    ''' check and convert to array
    Input:
    -- list or array (n, ), (n, 1), (1, n) 
    Output:
    -- (n, ) array '''

    if isinstance(input_variable, list):
        output = np.array(input_variable)
    elif isinstance(input_variable, (np.ndarray, np.generic)):
        # separate the length part, because scalar does not have length
        if len(input_variable.shape) == 1:
            output = input_variable
        else:
            output = input_variable.flatten()
    else:
        raise Exception('Please input 1-dim np.array or list')

    return output



