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
import matplotlib.pyplot as plt
import os,sys,inspect

# my module and functions
dirpath = os.getcwd()
while  (os.path.basename(dirpath) != "450-2-HW2") :
    dirpath = os.path.dirname(dirpath)

targetdir = dirpath + 'MCMC'

if targetdir not in sys.path:
    sys.path.insert(0,targetdir)


from MCMC import plot_process_distribution 

importlib.reload(plot_process_distribution)

# ---------------------------------------------------------------------------------
# class
# ---------------------------------------------------------------------------------


class mix_normal:
    """docstring for entry_likelihood"""

    def __init__(self, list_of_alphas= [], list_of_sigmas = [], p_mix = 0.5):


        self.alpha = list_of_alphas
        self.sigma = list_of_sigmas
        self.p_mix = p_mix
        np.random.seed(seed=13344)

        return 


    # I. the target distribution ------------------------------------------------------
    def _mix_normal_pdf(self, x):

        pdf = {}
        for i, alpha, sigma in zip(range(len(self.alpha)), self.alpha, self.sigma):
            pdf[i] = stats.multivariate_normal.pdf(x, mean=alpha,cov=sigma)
        
        pdf_values = pdf[0]*self.p_mix + pdf[1]*(1-self.p_mix)

        return pdf_values


    # II. generate markov chain  ----------------------------------------------------
    #     of which the steady state dist is the target distribution
    def get_a_process(self, x0, sigma, T=20, figsize=1):
        
        x_process = self._calculate_process(x0, sigma, T)

        N_newdraws = self._number_of_new_draws(x_process)
        print('number of new draws = {} (out of {})'.format(N_newdraws,T))

        fig = plot_process_distribution.plot_process(x_process, size = figsize)
        fig.show()

        fig = plot_process_distribution.contour_2dimrandomvar(x_process, size=figsize)
        fig.show()

        return x_process


    def _calculate_process(self, x0, sigma, T=20):
        
        x0 = np.array(x0)
        x = [x0]
        x_t = x0
        for i in range(T):
            x_t1 = self._get_x_prime(x_t, sigma)

            x = x + [x_t1]
            x_t= x_t1.copy()

        return np.array(x)

    def _get_x_prime(self, x, sigma):
        
        x_tilde = self._get_x_tilde(x, sigma)

        prob = self._calculate_prob_of_transaction(x_tilde,x,sigma)
        prob = min(1.0, prob)

        random_draw = np.random.choice([1, 0], size=(1,), p=[prob, 1-prob])
        x_prime = random_draw*x_tilde + (1-random_draw)*x

        return x_prime

    def _calculate_prob_of_transaction(self, x_tilde, x, sigma):

        f_x_tilde =self._mix_normal_pdf(x_tilde)
        f_x       =self._mix_normal_pdf(x)
        q_given_x =self._q_conditional_pdf(x_tilde,x,sigma)
        q_given_x_tilda = self._q_conditional_pdf(x,x_tilde,sigma)

        prob = (f_x_tilde*q_given_x_tilda) / (f_x*q_given_x)

        return prob

    def _q_conditional_pdf(self,x_tilde, x, sigma):
        '''Prob of x_tilde conditional on x '''

        epsilon = x_tilde - x
        q = stats.multivariate_normal.pdf(epsilon , mean=np.zeros(2), cov=np.eye(2)*sigma )

        return q


    def _get_x_tilde(self, x, sigma):

        epsilon = stats.multivariate_normal.rvs(mean=np.zeros(2), cov=np.eye(2)*sigma )

        x_tilde = x + epsilon 

        return x_tilde


    # II. auxiliary functions  ----------------------------------------------------
    def _number_of_new_draws(self,x):

        x_shift = np.roll(x, -1, axis = 0)
        update = (x[:-1] != x_shift[:-1]).astype(int)
        N = sum(np.sum(update, axis=1) > 0)

        return N


    def density_of_true_process(self, figsize = 1):

        size = figsize
        height = 7 * size 
        width = 10 * size 

        X, Y = np.mgrid[-6:6:100j, -6:6:100j]
        Z = np.zeros( (100,100) )
        for i in range(100):
            input_rv = np.vstack( (X[i,:], Y[i,:]) ).T
            Z[i,:] = self._mix_normal_pdf(input_rv)

        fig, ax = plt.subplots(figsize = (width,height))
        CS = ax.contour(X, Y, Z)
        #ax.clabel(CS, inline=1, fontsize=10)
        ax.set_title('Mixture Normal Density', fontsize = 20*size)
        fig.show()

        return 






