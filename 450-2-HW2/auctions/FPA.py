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

dirpath = os.getcwd()
while  (os.path.basename(dirpath) != "450-2-HW2") :
    dirpath = os.path.dirname(dirpath)

targetdir = dirpath + 'auctions'

if targetdir not in sys.path:
    sys.path.insert(0,targetdir)


from auctions import empirical_distributions 

importlib.reload(empirical_distributions)


# ---------------------------------------------------------------------------------
# class 
# ---------------------------------------------------------------------------------

class FPA_lognormal:
    """docstring for entry_likelihood"""

    def __init__(self, n_simul_draws = 1000):
        ''' 
        -- n_simul_draws is used to approx integrals, MUST BE LARGE ENOUGH。 
            Here I use the cdf function from scipy. But if the distribution is not normal, need to really draw the values and integral '''


        self.dist_par_sigma = np.sqrt(0.2)
        self.dist_par_mu = 1
        self._draw_sample_for_integral(n_simul_draws)



        return 

    # II. functions ----------------------------------------------------
    def get_bids(self, private_values , n_bidders ):
        '''private_values should be an array or list '''

        private_values = np.array(private_values)

        manipulation = self._get_manipulation(private_values, n_bidders=n_bidders)

        bids = private_values - manipulation

        return bids

    def get_rents(self, private_values, n_bidders ):
        '''rents = manipulation * prob_of_win, exactly the numerator of manipulation '''

        v_i = np.array(private_values)
        
        rents = []
        for i in range(len(v_i)):
            numerator = self._simulated_int_cdf(v_ub = v_i[i], n_bidders = n_bidders)
            rents = rents + [numerator]
        rents = np.array(rents)

        return rents

    def _get_manipulation(self, private_values, n_bidders ):

        v_i = np.array(private_values)

        manipulation = []
        for i in range(len(v_i)):
            numerator = self._simulated_int_cdf(v_ub = v_i[i], n_bidders = n_bidders)
            denominator = self._calculate_cdf(v_i[i])**(n_bidders-1)
            manipulation = manipulation + [numerator/denominator]
        manipulation = np.array(manipulation)

        return manipulation


    def simulate_auctions(self, n_auctions, n_bidders):

        
        # 1. simulate all bids
        all_bids = []
        all_pvs = []
        for i in range(n_bidders):
            pv_i = self._draw_private_values(n_auctions)
            bid_i = self.get_bids(pv_i, n_bidders = n_bidders)

            all_bids = all_bids + [bid_i]
            all_pvs = all_pvs + [pv_i]

        all_pvs = np.array(all_pvs)
        all_bids = np.array(all_bids)

        # 2. get the winner
        winning_bids = np.max(all_bids, axis = 0)

        return winning_bids, all_bids, all_pvs




    # III. import parameters and simulate data ----------------------------------------------------
    def _draw_private_values(self, n_auctions):

        sigma = self.dist_par_sigma
        mu = self.dist_par_mu

        v_i = stats.lognorm.rvs(s= sigma, loc = 0, scale = np.exp(mu), size=n_auctions*10)
        #v_i = np.random.lognormal(mean=1, sigma= 0.2, size=n_auctions)
        
        # v_i = np.clip(v_i, 0, 5)
        # the clip will cause a problem in estimation: there would be a peak in the pdf where pv=5, bids around the higest bids

        v_i = v_i[v_i<=5]
        v_i = v_i[:n_auctions]

        return v_i

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



    # IV. calculation functions ----------------------------------------------------
    def _calculate_cdf(self, v):

        x = self.int_simul_x_draws
        cdf_x = self.int_simul_cdf_draws

        cdf_lb = cdf_x[x<=v]
        if len(cdf_lb)!=0 :
            cdf_lb = cdf_lb[-1]
        else:
            cdf_lb = 0
        cdf_ub = cdf_x[x>v]
        if len(cdf_ub)!=0 :
            cdf_ub = cdf_ub[0]
        else:
            cdf_ub = 1


        return (cdf_lb+cdf_ub)/2

    def _simulated_int_cdf(self, v_ub, n_bidders):
        '''' integral of cdf^{n-1} dx, from 0 to upper bound v_ub. '''

        # ---- 1. get x and corresponding cdfs
        x = self.int_simul_x_draws
        cdf_x = self.int_simul_cdf_draws

        cdf_x = cdf_x[x<=v_ub]
        x = x[x<=v_ub]

        # ---- 2. integral
        integral =  sum(cdf_x**(n_bidders-1))/len(x)

        #integral_tocheck_should_eq_expectation = sum(1-cdf_x)/self.n_simul_draws

        return integral



# ---------------------------------------------------------------------------------
# class 2 
# ---------------------------------------------------------------------------------

class estimate_FPA:


    def __init__(self, n_bidders, winning_bids = None, all_bids = None, true_pvs = None):

        self.n_bidders = n_bidders
        self.obs_winning_bids = winning_bids
        self.obs_all_bids = all_bids
        self.true_pvs = true_pvs

        return 




    # I. back out true values ----------------------------------------------------
    def back_out_private_values_pointbypoint_fromallbids(self):

        if self.obs_all_bids is not None:
            # ---- 1. empirical distribution of all bids
            self._get_empirical_bids_distribution()

            length, width = self.obs_all_bids.shape
            all_bids = self.obs_all_bids.reshape( (length*width,) ) 
            cdf_values = self.bids_dist.cdf( all_bids )
            pdf_values = self.bids_dist.pdf( all_bids )

            # ---- 2. back out true values
            true_pvs = all_bids + 1/(self.n_bidders - 1) * cdf_values/pdf_values 

            self.est_pvs =  true_pvs.reshape( (length, width) )
            return 

        else:

            print('Do not observe all bids. Can not approx cdf')
            return

    def back_out_private_values_pointbypoint_fromwinningbids(self):

        if self.obs_winning_bids is not None:

            # ---- 1. empirical distribution of all bids
            self._get_empirical_bids_distribution(all_bids=False)

            winning_bids = self.obs_winning_bids
            cdf_values = self.winning_bids_dist.cdf( winning_bids )
            pdf_values = self.winning_bids_dist.pdf( winning_bids )

            # adjust (because the above is the winning distribution)
            cdf_values = cdf_values**(1/self.n_bidders)
            pdf_values = pdf_values / (self.n_bidders * (cdf_values**(self.n_bidders-1) ) )

            # ---- 2. back out true values
            true_pvs = winning_bids + 1/(self.n_bidders - 1) * cdf_values/pdf_values 

            self.est_pvs =  true_pvs
            return 

        else:

            print('Do not have winning bids data')
            return


            
    # II. get the distribution ----------------------------------------------------
    def estimate_private_value_distribution(self, from_all_bids = True):

        if not hasattr(self, 'est_pv'):
            if from_all_bids:
                self.back_out_private_values_pointbypoint_fromallbids()
            else:
                self.back_out_private_values_pointbypoint_fromwinningbids()
        
        self.pv_dist = empirical_distributions.rv_1D(self.est_pvs)
        
        return


    # III. other functions  ----------------------------------------------------
    def _get_empirical_bids_distribution(self, all_bids = True):

        if all_bids:
            if self.obs_all_bids is not None:
                self.bids_dist = empirical_distributions.rv_1D(self.obs_all_bids)
            else:
                print('Do not observe all bids. Can not approx cdf')
            return
        else:
            if self.obs_winning_bids is not None:
                self.winning_bids_dist = empirical_distributions.rv_1D(self.obs_winning_bids)
            else:
                print('Do not observe all bids. Can not approx cdf')
            return









