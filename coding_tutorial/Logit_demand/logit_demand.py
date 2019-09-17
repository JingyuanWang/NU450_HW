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
    # I: vector of consumer index, n*c-by-1
    # X: matrix of consumer choice attributes, n*c-by-m, index = consumer_id
    # beta: coefficients, m-by-1
    
    # 1. check input data ----------------------------------------------------------------------
    # suppose the column size of X is correct
    nc,k = X.shape 
    # check X
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if not isinstance(X, pd.DataFrame):
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
    
    # check index I
    if I.shape != (nc,1) and I.shape != (nc,):
        raise Exception("size of I should be ({},1) or ({},) instead of {}".format(nc, nc, I.shape))

    # 2. calculate the probability --------------------------------------------------------------
    # (0) set up a dataframe to work with
    df = X.copy()
    df = df.reset_index()
    df['consumer_id'] = I
    # df = df.set_index('consumer_id')
    # (1). give a score to each choice for each consumer
    df['score'] = X.values.dot(beta)
    # later on we will do \[prob = exp(score)/sum( exp(score)  of all product offered) \] for each person
    # to avoid exp(something large) = inf:
    #     for each person, 
    #         devide the denominator and numerator by exp(max product score for this person)
    #         equivalent to 
    #         score for each product - max score 
    for consumer, frame in df.groupby('consumer_id'):
        max_index = np.max(frame['score'])
        df.loc[df['consumer_id'] == consumer,'score'] = frame['score'] - max_index
    
    # (2). calculate a probability of chosing each product for each consumer, based on the score
    for consumer, frame in df.groupby('consumer_id'):
        total_expscore_thisperson = np.sum(np.exp(frame['score']))
        df.loc[df['consumer_id'] == consumer,'prob'] = frame.apply(lambda row: np.exp(row['score'])/total_expscore_thisperson, 
                                                               axis = 1)
    
    # 3. check and return values ---------------------------------------------------------------
    # check whether prob for each person sum up to 1
    total_prob = df.groupby('consumer_id').agg({'prob': np.sum})
    tol = 10**(-10)
    if len(total_prob.loc[abs(total_prob['prob'] -1) > tol]) != 0:
        print("probability does not sum up to 1, return the dataframe")
    
    probability = df['prob']
    if len(probability) != len(I):
        raise Exception("size of probability should be the same as I {}-by-1, current value {}-by-1".format(I.shape,I.shape))   

    return probability
    # return a vector of probablities, same size as I (n*k-by-1)


def loglikelihood(Y,I,X,beta):
    # Purpose: compute the NEGATIVE likelihood of a binary vector of choices Y, conditional on X
    # Inputs:
    # Y: binary-vector of choices, n*c-by-1
    # I: vector of consumer index, n*c-by-1
    # X: matrix of consumer choice attributes, n*c-by-m
    # beta: coefficients, m-by-1
    
   
    # 1. check input data ----------------------------------------------------------------------
    # suppose the column size of X is correct
    nc,k = X.shape 
    # check X
    if not np.logical_or(isinstance(X, pd.DataFrame), isinstance(X, np.ndarray)):
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
    if Y.shape != (nc,1) and Y.shape != (nc,):
        raise Exception("size of Y should be ({},1) or ({},) instead of {}".format(nc, nc, Y.shape))
    
    # check index I
    if I.shape != (nc,1) and I.shape != (nc,):
        raise Exception("size of I should be ({},1) or ({},) instead of {}".format(nc, nc, I.shape))

    
    # 2. calculate loglikelihood -------------------------------------------------------------------
    precision = 10**-300
    
    # 2.1 likelihood for each obs = consumer-product
    likelihood_c_j = choice_probability(I,X,beta)
    
    # 2.2 likelihood for each consumer to make the observed choice
    likelihood_c_j.reset_index(drop = True, inplace = True)
    likelihood_c = likelihood_c_j.loc[Y==1]
    
    # [Adjustment 1] Avoid -inf:
    # If there is a choice probability almost = 0.0
    #         then the loglikelihood for this obs = -inf
    # then the summation of all obs == -inf, no matter how the other obs performs. This obs is weighted 100%.
    # to avoid this, remove all the 0s and replace to the precision or the minimum nonzero value.
    likelihood_c[likelihood_c<=precision] = precision
    
    # 2.3 summation of loglikelihood for all obs
    loglikelihood = -np.sum(np.log(likelihood_c))
    

    # 3. return ------------------------------------------------------------------------------------
    return loglikelihood
    # return a number, the NEGATIVE likelihood y|x



