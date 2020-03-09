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




class rv_1D:


    def __init__(self, observations):
        '''observations should be a list or array'''
        
        self.observations = observations.ravel()
        self.kde          = stats.gaussian_kde(self.observations)

        return


    def cdf(self, x):
        '''x can be a single number, a list or array '''

        if isinstance(x,float) or isinstance(x, int):
            cdf_values = self.kde.integrate_box_1d(0,x)
        else:
            cdf_values = []
            for i in x:
                cdf = self.kde.integrate_box_1d(0,i)
                cdf_values = cdf_values + [cdf]
            cdf_values = np.array(cdf_values)

        return cdf_values

    def pdf(self,x):

        pdf_values = self.kde.pdf(x)

        return pdf_values



# the below class is for other research and future uses:

class rv_2D:

    
    def __init__(observations):
        '''observations should be a list or array, 2-by-n'''
        
        self.observations = observations

        return


    def calculate_empirical_joint_cdf(self, x):
        '''x should be a tuple/list/array of 2-dimensions '''
    

        return cdf_values

    def calculate_empirical_marginal_cdf_1(self,x):


        return cdf_values

    def calculate_empirical_marginal_cdf_2(self,x):


        return cdf_values




