import numpy as np
import pandas as pd
import scipy.stats as stats

# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose:
# define functions
#
# Definition of several variables in this file:
# n: number of consumers
# m: number of products
#    outside option normalized to 0 and not included among the choices
# k: length of beta, number of variables
# ------------------------------------------------------------------------

def choice_probability(I,X,beta):
    # Purpose: compute the probability of each consumer making each choices
    # Inputs:
    # I: vector of consumer index, n*m-by-1
    # X: matrix of consumer choice attributes, n*m-by-k
    # beta: coefficients, k-by-1
    prob = I
    
    return prob
    # return a vector of probablities, same size as I (n*k-by-1)


def loglikelihood(Y,I,X,beta):
    # Purpose: compute the NEGATIVE likelihood of a binary vector of choices Y, conditional on X
    # Inputs:
    # Y: binary-vector of choices, n*m-by-1
    # I: vector of consumer index, n*m-by-1
    # X: matrix of consumer choice attributes, n*m-by-k
    # beta: coefficients, k-by-1
    
    # 0. set parameters ------------------------------------------------------------------------
    # assume the epsilon ~ N(0,sigma)
    sigma = 30
    
    
    # 1. check input data ----------------------------------------------------------------------
    # suppose the size of X is correct
    nm,k = X.shape 
    # check X
    if isinstance(X, pd.DataFrame):
        X = X.values
    if not isinstance(X, np.ndarray):
        raise Exception("X should be a dataframe or an array")
    
    # check beta
    # (1) data type
    if isinstance(beta,list) or isinstance(beta,pd.Series):
        beta = np.array(beta)
    if not isinstance(beta, np.ndarray):
        raise Exception("beta should be a list, a series, or an array")
    # (2) check the size of beta = (k,1)
    if beta.shape == (k,):
        beta = beta.reshape((k,1))
    if beta.shape != (k,1):
        raise Exception("size of beta should be ({},1) or ({},) instead of {}".format(k, k, beta.shape))
    
    # check Y  
    if Y.shape != (nm,1) and Y.shape != (nm,):
        raise Exception("size of Y should be ({},1) or ({},) instead of {}".format(nm, nm, Y.shape))
    
    # check index I
    if I.shape != (nm,1) and I.shape != (nm,):
        raise Exception("size of I should be ({},1) or ({},) instead of {}".format(nm, nm, I.shape))
    
    
    # 2. calculate loglikelihood -------------------------------------------------------------------
    precision = 10**-100
    
    # 2.1 likelihood for each obs
    likelihood_y_1=pd.Series(map(lambda x: (1-stats.norm.cdf(x, loc=0, scale=sigma)), 
              X.dot(beta)*(-1) ))

    likelihood_y_0=pd.Series(map(lambda x: stats.norm.cdf(x, loc=0, scale=sigma), 
              X.dot(beta)*(-1) ))
    
    # [Adjustment 1] Avoid -inf:
    # If there is some beta s.t. 
    #         beta*x >p99.99 of N(0,sigma), then likelihood = 0.0, loglikelihood for this obs = -inf
    # then the summation of all obs == -inf, no matter how the other obs performs. This obs is weighted 100%.
    # to avoid this, remove all the 0s and replace to the precision or the minimum nonzero value.
    likelihood_y_1[likelihood_y_1==0] = min(precision, likelihood_y_1[likelihood_y_1!=0].min()) 
    likelihood_y_0[likelihood_y_0==0] = min(precision, likelihood_y_0[likelihood_y_0!=0].min()) 
    
    # 2.2 summation of loglikelihood for all obs
    loglikelihood = -np.sum(list(map(lambda y, p1, p0: y*np.log(p1) + (1-y)*np.log(p0),
                                     Y,likelihood_y_1, likelihood_y_0)))
    
    
    # 3. return ------------------------------------------------------------------------------------
    return loglikelihood
    # return a number, the NEGATIVE likelihood y|x



