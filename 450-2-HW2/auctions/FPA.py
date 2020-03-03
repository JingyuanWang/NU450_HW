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

class FPA_lognormal:
    """docstring for entry_likelihood"""

    def __init__(self, n_samples= 100, n_simul_draws = 1000):


        self.dist_par_sigma = np.sqrt(0.2)
        self.dist_par_mu = 1
        
        self._draw_private_values(n_samples)
        self._draw_sample_for_integral(n_simul_draws)


        return 


    # I. import parameters and simulate data ----------------------------------------------------
    def _draw_private_values(self, n_samples):

        sigma = self.dist_par_sigma
        mu = self.dist_par_mu

        v_i = stats.lognorm.rvs(s= sigma, loc = 0, scale = np.exp(mu), size=n_samples)
        #v_i = np.random.lognormal(mean=1, sigma= 0.2, size=n_samples)
        v_i = np.clip(v_i, 0, 5)
        self.v_i = v_i
        self.n_samples=  n_samples

        return

    def _draw_sample_for_integral(self, n_simul_draws):

        self.n_simul_draws = n_simul_draws
        sigma = self.dist_par_sigma
        mu = self.dist_par_mu

        # ---- 1. get valus
        self.int_simul_x_draws = np.arange(0,n_simul_draws*5)/n_simul_draws

        # --- 2. get cdf(x)
        normalize_for_truncation = stats.lognorm.cdf(x=5, s = sigma, scale = np.exp(mu), loc = 0)
        cdf_x = stats.lognorm.cdf(self.int_simul_x_draws, s = sigma, scale = np.exp(mu), loc = 0)
        cdf_x = cdf_x/normalize_for_truncation
        self.int_simul_cdf_draws = cdf_x

        return

    def get_bids(self, n_bidders = 2):
        '''the input should be an array '''

        v_i = self.v_i

        manipulation = []
        for i in range(self.n_samples):
            numerator = self._simulated_int_cdf(v_ub = v_i[i], n_bidders = n_bidders)



        return 

    def _calculate_cdf(self, v):

        return

    def _simulated_int_cdf(self, v_ub, n_bidders):
        '''' integral of cdf^{n-1} dx, from 0 to upper bound v_ub. '''

        # ---- 1. get x and corresponding cdfs
        x = self.int_simul_x_draws
        cdf_x = self.int_simul_cdf_draws

        cdf_x = cdf_x[x<=v_ub]
        x = x[x<=v_ub]

        # ---- 2. integral
        integral =  sum(cdf_x**(n_bidders-1))/self.n_simul_draws

        #integral_tocheck_should_eq_expectation = sum(1-cdf_x)/self.n_simul_draws

        return integral










