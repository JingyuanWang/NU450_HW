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
        #u = np.random.normal(0, 1, (F, n_sample) )
        #self.u0 = u

        #xi = np.random.lognormal(0, 1, (F, n_sample) )
        #self.xi = xi

        return

    def _get_data(self):

        # (1). get parameters
        M = self.M
        F = self.F 
        #u0 = self.u0
        #n_sample = u0.shape[1]

        # (2) get data: long, each row = a firm (instead of a mkt)
        pi = self.df[ [x for x in self.df.columns if x.startswith('pi')] ].values
        Z  = self.df[ [x for x in self.df.columns if x.startswith('Z')] ].values
        X  = self.df[ [x for x in self.df.columns if x.startswith('X')] ].values
        N  = self.df[ [x for x in self.df.columns if x.startswith('N')] ].values
        entry = self.df[ [x for x in self.df.columns if x.startswith('entry')] ].values
        xi    = self.df[ [x for x in self.df.columns if x.startswith('xi')] ].values

        return (M, F, pi, Z, X, N, xi, entry)

    def _get_data_long(self):

        (M, F, pi, Z, X, N, xi, entry) = self._get_data()

        pi = pi.reshape( (M*F,1), order= 'C' )
        Z = Z.reshape( (M*F,1), order= 'C' )
        entry = entry.reshape( (M*F,1), order= 'C' )
        X = np.kron(X, np.ones((F,1) ))
        N = np.kron(N, np.ones((F,1) ))     
        xi = np.kron(xi, np.ones((F,1) ))      

        df = pd.DataFrame()
        df['N'] = N[:,0]
        df['N_potential'] = N[:,1]
        df['X'] = X
        df['xi'] = xi
        df['pi'] = pi
        df['Z'] = Z
        df['entry'] = entry

        return df









