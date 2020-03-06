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




# ---------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------

def plot_process(x, size = 1):

    height = 7 * size 
    width = 10 * size 
    legend_font_size = 16 * size 
        
    fig = plt.figure( figsize = (width, height))

    plt.plot(x[:,0], color='maroon')
    plt.plot(x[:,1], color='orange')

    return fig


def contour_2dimrandomvar(x, size = 1):


    height = 7 * size 
    width = 10 * size 

    # ---- 1. read in values
    if x.shape[1] != 2:
        if x.shape[0] == 2:
            x = x.T 
        else:
            raise Exception('x is not 2 dimensional')  

    m1=x[:,0]
    m2=x[:,1]    

    # ---- 2. set the aixs
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()    

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    # ---- 3. calculate density
    values = np.vstack([m1, m2])    
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)    

    # ---- 4. plot
    fig, ax = plt.subplots(figsize = (width,height))
    CS = ax.contour(X, Y, Z)
    #ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Simulated Density', fontsize = 20*size)

    #plt.close()

    return fig






