'''
# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose:
# define class: 
#
# ------------------------------------------------------------------------
'''

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt
from scipy.stats import norm
import importlib


class entry_likelihood:
    """docstring for entry_likelihood"""

    def __init__(self, M,F,df, n_sample= 1000):
        
        self._import_data(df)

        self._setting_parameters(M,F)


        # prepare for estimation
        self._initialize(n_sample)

        return 


    # I. import parameters and data ----------------------------------------------------
    def _import_data(self,df):

        self.df = df

        return

    def _setting_parameters(self, M, F):
        '''Get setting parameters: number of markets, number of potential entrants '''

        self.M = M 
        self.F = F

        return 

    # II. Prepare for estimation --------------------------------------------------------
    def _initialize(self, n_sample):
        '''Fix error term of simulation '''

        # 1. get parameters
        M = self.M
        F = self.F 

        # number of simulation draws in integral
        u = np.random.normal(0, 1, (F, n_sample) )
        self.u0 = u

        return

    def _get_data(self):

        # (1). get parameters
        M = self.M
        F = self.F 
        u0 = self.u0
        n_sample = u0.shape[1]

        # (2) get data
        Z = self.df[ ['Z_1m', 'Z_2m', 'Z_3m'] ].values
        X = self.df[ ['X_m'] ].values
        N = self.df[ ['N_m'] ].values

        return (M,F, u0,Z, X,N, n_sample)

    # II calculate likelihood -----------------------------------------------------------
    def get_likelihood(self, alpha, beta, delta, mu, sigma):

        (M,F,_,_,_,_,n_sample) = self._get_data()

        # 1. get the outcome
        entry = self.df[ ['entry_1m', 'entry_2m', 'entry_3m'] ].values 
        entry = entry.reshape( (M*F,1), order = 'C' )

        # 2. simulation : satisfies eqm outcomes (observable) & assumptions

        # (1) enter or not: 300*500, the first 3 rows are the 3 firms in the first market
        (simul_enter, simul_notenter) = self._simul_entry_outcome(alpha, beta, delta, mu, sigma)

        # (2) enter order assumptions
 
        

        # 3. likelihood
        prob = np.sum((entry*simul_enter) + (1-entry)*simul_notenter , axis = 1)/10000

        precision = 10**-320  # np.log(10**-330) = -inf
        prob[prob<=precision] = precision
        
        likelihood = np.sum( np.log(prob) )

        return likelihood

    def _simul_entry_order_lowestfirst(self, alpha, mu, sigma):
        '''Output:
        -- simul_ordering: 100*500, simulate whether the market satisfies the assumed ordering '''

        # eqm outcome
        M = 100
        F = 3
        entry = self.df[ ['entry_1m', 'entry_2m', 'entry_3m'] ].values 
        entry = entry.reshape( (M*F,1), order = 'C' )

        # firm cost random draw: 300*500, the first 3 rows are the 3 firms in the first market
        phi_fm = self._get_firm_specific_cost(alpha, mu, sigma)

        
        #entry * phi_fm


        return entry, phi_fm
         #return simul_ordering

    def _simul_entry_outcome(self, alpha, beta, delta, mu, sigma):
        '''Fix error term of simulation '''
        
        (_,F,_,_,_,_,_) = self._get_data()

        # 1. current market condition: 100*1
        (pi_market_enter, pi_market_notenter) = self._get_mkt_level_profit(beta, delta)

        # 2. firm random draw: 300*500, the first 3 rows are the 3 firms in the first market
        phi_fm = self._get_firm_specific_cost(alpha, mu, sigma)

        # 3. get probabilities 300*500, the first 3 rows are the 3 firms in the first market
        simul_enter = phi_fm < np.kron(pi_market_enter, np.ones( (F,1) ) ) 
        simul_notenter = phi_fm > np.kron(pi_market_notenter, np.ones( (F,1) ) ) 

        return (simul_enter, simul_notenter)

    def _get_mkt_level_profit(self, beta, delta):
        '''Output:  
        -- pi_market_enter: 100*1, each row is a market, market level profit for entrant
        -- pi_market_notenter:  100*1, each row is a market, market level profit for potential entrant '''

        (_,_,_,Z,X,N,_) = self._get_data()

        # market level profit for the entrant and the potential entrant, 100*1, each row is a market
        pi_market_enter = X*beta + np.log(N)*delta 
        pi_market_notenter = X*beta + np.log(N+1)*delta 

        return (pi_market_enter, pi_market_notenter)

    def _get_firm_specific_cost(self, alpha, mu, sigma):
        '''Output:  
        -- phi_fm: realization of fixed cost: 300 * 500, the first 3 rows are the 3 firms in the first market '''

        (M,F,u0,Z,_,_,_) = self._get_data()
        # cost realization
        u = mu + sigma*self.u0
        
        # realization of fixed cost: 300 * 500, the first 3 rows are the 3 firms in the first market
        phi_fm =  alpha*Z.reshape( (M*F,1) , order = 'C' ) + np.kron( np.ones( (M,1) ) , u )

        return phi_fm












