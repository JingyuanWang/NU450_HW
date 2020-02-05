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
    def get_likelihood(self, alpha, beta, delta, mu, sigma, order = None):

        (M,F,_,_,_,_,n_sample) = self._get_data()

        # 1. get the outcome
        entry = self.df[ ['entry_1m', 'entry_2m', 'entry_3m'] ].values 
        entry = entry.reshape( (M*F,1), order = 'C' )

        # 2. simulation : satisfies eqm outcomes (observable) & assumptions

        # (1) enter or not: 300*5000, the first 3 rows are the 3 firms in the first market
        (simul_enter, simul_notenter) = self._simul_entry_outcome(alpha, beta, delta, mu, sigma)

        # (2) enter order assumptions
        if order == 'lowestfirst':
            # firm random draw: 300*5000, the first 3 rows are the 3 firms in the first market
            phi_fm = self._get_firm_specific_cost(alpha, mu, sigma)
            simul_order = self._simul_entry_order_lowestfirst(M,F, n_sample, entry, phi_fm)
        if order == 'highestfirst':
            # firm random draw: 300*5000, the first 3 rows are the 3 firms in the first market
            phi_fm = self._get_firm_specific_cost(alpha, mu, sigma)
            simul_order = self._simul_entry_order_highestfirst(M,F, n_sample, entry, phi_fm)
        
        # 3. likelihood
        if order == None:
            prob = np.sum((entry*simul_enter) + (1-entry)*simul_notenter , axis = 1)/n_sample

            # equivalent
            #simul_market_level = np.prod(((entry*simul_enter) + (1-entry)*simul_notenter).reshape(M,F,n_sample), axis = 1)
            #prob = np.sum(simul_market_level , axis = 1)/n_sample

        else:
            simul_market_level = np.prod(((entry*simul_enter) + (1-entry)*simul_notenter).reshape(M,F,n_sample), axis = 1)
            simul_market_level = simul_market_level*simul_order
            prob = np.sum(simul_market_level , axis = 1)/n_sample


        precision = 10**-320  # np.log(10**-330) = -inf
        prob[prob<=precision] = precision
        
        likelihood = np.sum( np.log(prob) )

        return likelihood

    def _simul_entry_order_lowestfirst(self, M,F, n_sample, entry, phi_fm):
        '''Output:
        -- simul_ordering: 100*5000, simulate whether the market satisfies the assumed ordering '''

        # get the maximum cost of firms that entered
        costs_enter = (entry * phi_fm)
        costs_enter[costs_enter == 0] = np.nan
        max_phi_entered = np.max( costs_enter.reshape( (M,F,n_sample) ), 
                                 where=~np.isnan(costs_enter.reshape( (M,F,n_sample) )), 
                                 axis = 1 , 
                                 initial = -9999)        

        # get the minimum cost of firms that did not enter
        costs_notenter = (1-entry) * phi_fm
        costs_notenter[costs_notenter == 0] = np.nan
        min_phi_notenter = np.min( costs_notenter.reshape( (M,F,n_sample) ), 
                                 where=~np.isnan(costs_notenter.reshape( (M,F,n_sample) )), 
                                 axis = 1 , 
                                 initial = 9999)        

        simul_order = max_phi_entered < min_phi_notenter


        return simul_order

    def _simul_entry_order_highestfirst(self, M,F, n_sample, entry, phi_fm):
        '''Output:
        -- simul_ordering: 100*5000, simulate whether the market satisfies the assumed ordering '''

        # get the maximum cost of firms that entered
        costs_enter = (entry * phi_fm)
        costs_enter[costs_enter == 0] = np.nan
        min_phi_entered = np.min( costs_enter.reshape( (M,F,n_sample) ), 
                                 where=~np.isnan(costs_enter.reshape( (M,F,n_sample) )), 
                                 axis = 1 , 
                                 initial = 9999)        

        # get the minimum cost of firms that did not enter
        costs_notenter = (1-entry) * phi_fm
        costs_notenter[costs_notenter == 0] = np.nan
        max_phi_notenter = np.max( costs_notenter.reshape( (M,F,n_sample) ), 
                                 where=~np.isnan(costs_notenter.reshape( (M,F,n_sample) )), 
                                 axis = 1 , 
                                 initial = -9999)        

        simul_order = min_phi_entered > max_phi_notenter

        return simul_order
 

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












