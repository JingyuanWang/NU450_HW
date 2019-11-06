'''
# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose:
# define class: Discrete Choice
#
# methods include:
# I. simulate consumer choices
# II. estimate coeff using BLP
# ------------------------------------------------------------------------
'''


import numpy as np
import pandas as pd
import os,sys,inspect
import scipy.stats as stats
import scipy.integrate as integrate
import itertools as it
import copy
import importlib


# random seed
np.random.seed(seed=13344)



# ---------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------

class BLP_MPEC:
    '''Samples for Discrete Choice Demand system estimation
    main part: 
    -- 3 dataframs, consumers, products, consumer_product
    -- functions: 
       -- stats
       -- simulation
       -- estimation '''

    # get the main dataframe (and get the set of variables)
    def __init__(self, DiscreteChoice):
        '''initialize a dataclass for discrete demand system analysis
        
        Input:
        -- DiscreteChoice, classed defined in the DiscreteChoice module in the same folder '''
        
        # get the data: only need the products attributes. In practise, we know nothing about the consumer.
        self.products = DiscreteChoice.products.sort_values(['market_id','product_id']) # must sorted this way

        # parameters
        self.num_of_market = DiscreteChoice.num_of_market
        self.num_of_prod = DiscreteChoice.num_of_prod

        # prepare for MPEC estimation
        self.prepare_for_MEPC()

    # I. basic functions ----------------------------------------------------

    def prepare_for_MEPC(self, n_sample = 5000):
        '''Prepare  for MEPC:
        1. draw random sample
        2. save a matrix of 0s and 1s  

        Input:
        -- n_sample: number of draws in the simulated integral part.
        '''

        # 1. get parameters 
        num_of_market = self.num_of_market
        num_of_prod = self.num_of_prod


        # 2. Randomly draw a sample of random taste
        v = np.random.lognormal(0, 1, (1, n_sample) )
        # save
        self.MEPC_par_n_sample = n_sample
        self.MPEC_par_sample_v = v

        # 3. construct a matrix (of 1s and 0s) with cluster diagnals 
        one_markect = np.ones((num_of_prod,num_of_prod))
        market_cluster = np.kron(np.eye(num_of_market), one_markect)
        # save
        self.MEPC_par_market_block_matrix = market_cluster


    # III. Generate Variables --------------------------------------------------------------------

    # ------------------------------------------
    # function group 1: 
    # generate exogeneous variables
    # ------------------------------------------
    def construct_exogenous_var(self):

        # save exogenous var names
        self.exogeneous_var = ['x1', 'x2', 'x3']

        # generate IV
        self.gen_BLP_instruments()
        self.gen_Hausman_instrument()


        return
    
    def gen_BLP_instruments(self):

        df = self.products

        func1 = lambda series: pd.DataFrame({'rolling': np.roll(series.values, 1).tolist() })
        func2 = lambda series: pd.DataFrame({'rolling': np.roll(series.values, 2).tolist() })    

        df['x2_other1'] = df.groupby('market_id')['x2'].apply(func1).reset_index(drop = True)
        df['x2_other2'] = df.groupby('market_id')['x2'].apply(func2).reset_index(drop = True)
        df['x3_other1'] = df.groupby('market_id')['x3'].apply(func1).reset_index(drop = True)
        df['x3_other2'] = df.groupby('market_id')['x3'].apply(func2).reset_index(drop = True)

        # save
        self.products = df.sort_values(['market_id','product_id']).reset_index(drop = True)
        self.exogeneous_var_BLPinstruments = ['x2_other1', 'x2_other2' , 'x3_other1' , 'x3_other2']

        return 

    def gen_Hausman_instrument(self):

        df = self.products
        num_of_market = self.num_of_market

        p_sum = df.groupby('product_id').agg({'price': np.sum }).rename(columns = {'price': 'price_others'} )
        df = pd.merge(df, p_sum, left_on = 'product_id', right_on = 'product_id')

        df['price_others'] = (df['price_others'] - df['price']) / (num_of_market-1)

        # save
        self.products = df.sort_values(['market_id','product_id']).reset_index(drop = True)
        self.exogeneous_var_Hausmaninstruments = ['price_others']




    # III. Estimate --------------------------------------------------------------------

    # ------------------------------------------
    # function group 1: 
    # 1. given delta, get shares, use build in integral
    # 2. given shares, back out delta: contraction mapping using 1
    # ------------------------------------------

    def products_social_ave_valuation_TO_market_share(self, delta, sigma_p , price_varname, for_gradiant = False, worried_about_inf = False):
        '''A function calculate market share using given (product social value) delta, using simulated integral 
        
        Input:
        --n_sample: number of draws in the simulated integral
        '''
        
        # 0. Prepare
        # take data of interest:
        df = self.products.copy()
        df = df.sort_values(['market_id','product_id']).reset_index(drop = True)
        price = df[price_varname].values[:,np.newaxis]
        
        # set parameter
        num_of_market = self.num_of_market
        num_of_prod = self.num_of_prod
        n_sample = self.MEPC_par_n_sample

        # the matrix (of 1s and 0s) with cluster diagnals 
        market_cluster = self.MEPC_par_market_block_matrix

        # 1. Randomly draw a sample of random taste
        v = self.MPEC_par_sample_v
        v_all = np.repeat(v, num_of_market*num_of_prod, axis = 0) # to vectorize the simulation

        # 2. Calculate Each Consumer's Preference
        # --1 get the social average valuation
        delta_all = np.repeat(delta, n_sample, axis = 1)

        # --2 random taste
        random_taste = - price * v_all * sigma_p

        # --3 score on each product
        score_all = np.exp(random_taste) * np.exp(delta) 

        # --4 probability of chosing each product
        total_score_all = (market_cluster@score_all) 
        share_all = score_all / (total_score_all + 1)

        if worried_about_inf:
            score_all = random_taste + delta
            score_separate_market = np.reshape(score_all, 
                                                [num_of_market,num_of_prod,n_sample])   
            # within in dim0, each matrix's shape is num_prod by n_sample, a market            

            market_maxscore = np.max(score_separate_market, axis = 1)   
            # within each market, take the maxscore for each person (each column)            

            score_all = score_all - np.repeat(market_maxscore, 3, axis = 0)
            outside = - np.repeat(market_maxscore, 3, axis = 0)
            
            # --3 score on each product
            score_all = np.exp(score_all)
            outside = np.exp(outside)
            
            # --4 probability of chosing each product
            total_score_all = (market_cluster@score_all) 
            share_all = score_all / (total_score_all + outside)

        if for_gradiant:
            # return a matrix of num_of_market*num_of_prod by number_of_consumers, about 300 -by- 5000
            return {'share_all': share_all, 'price' : price}
        else:
            # 3. Integral: average across all simulated values
            shares = (np.sum(share_all, axis = 1) / n_sample)
            shares = shares[:,np.newaxis]
            return shares


    # ------------------------------------------
    # function group 2: 
    # objective functions and constrains
    # ------------------------------------------

    # ------------------------------------------
    # function group 3: 
    # gradient for objective function
    # ------------------------------------------

    # ------------------------------------------
    # function group 4: 
    # gradient for constraint
    # ------------------------------------------
    def gradiant_share(self, delta, sigma_p , price_varname):
        '''derivatives of the share function (a set of constraints, num_of_prod * num_of_market, denoted JM) 

        Output:
        -- derivatives, (JM+1) -by- JM. 
                        each column is the derivative of one constraint on 
                               -- JM deltas (social average valuation on each product-market);
                               -- 1 sigma (taste randomness)
        '''
        interim_integral_results = self.products_social_ave_valuation_TO_market_share(delta, sigma_p , price_varname, for_gradiant = True)
        share_all = interim_integral_results['share_all']
        price     = interim_integral_results['price']

        derivatives_firstJMrows = self.gradiant_share_on_social_ave_valuation(share_all)

        derivatives_laterrows = self.gradiant_share_on_taste_randomness(share_all,price)
        
        gradiant = np.vstack((derivatives_firstJMrows, derivatives_laterrows))

        return gradiant

    def gradiant_share_on_social_ave_valuation(self, share_all):
        '''derivatives of the share function (a set of constraints, num_of_prod * num_of_market, denoted JM) 
        Partially on deltas (social average valuation)

        Output:
        -- derivatives, JM -by- JM. each column is the derivative of one constraint on JM deltas (social average valuation on each product-market).
        '''

        # the matrix (of 1s and 0s) with cluster diagnals 
        market_cluster = self.MEPC_par_market_block_matrix
        n_sample = self.MEPC_par_n_sample

        # derivatives
        derivatives = - share_all@share_all.T / n_sample
        # @ product did the Integral!

        derivatives = derivatives * market_cluster
        # off diagnal blocks = 0

        # adjust for the diagnals
        shares = (np.sum(share_all, axis = 1) / n_sample)
        shares_1dim = np.squeeze(shares) # num_of_prod * num_of_market - by - 1 , to num_of_prod * num_of_market - by - 0
        derivatives_diagnal = np.diag(shares_1dim)  # num_of_prod * num_of_market - by - num_of_prod * num_of_market
        derivatives = derivatives_diagnal + derivatives
        
        return derivatives

    def gradiant_share_on_taste_randomness(self, share_all, price):
        '''derivatives of the share function (a set of constraints, num_of_prod * num_of_market, denoted JM) 


        Output:
        -- derivatives, 1 -by- JM. each column is the derivative of one constraint on sigma (taste randomness).
        '''

        # 1. GET DATA AND PARAMETERS
        # the matrix (of 1s and 0s) with cluster diagnals 
        market_cluster = self.MEPC_par_market_block_matrix
        n_sample = self.MEPC_par_n_sample
        num_of_market = self.num_of_market
        num_of_prod = self.num_of_prod
        
        # the random sample
        v = self.MPEC_par_sample_v
        v_all = np.repeat(v, num_of_market*num_of_prod, axis = 0) # to vectorize the simulation

        # derivative of the numerator 
        derivative_numerator_all =  share_all * price * v_all

        # derivative of the denominator
        partial_all = share_all * price * v_all
        partial_sum_all = (market_cluster@partial_all) 
        derivative_denominator_all = - share_all * partial_sum_all

        # derivative
        derivative_all = derivative_numerator_all + derivative_denominator_all
        derivative = (np.sum(derivative_all, axis = 1) / n_sample)
        
        # each row stands for 1 parameter; each columns stands for 1 constraint.
        derivative = derivative[np.newaxis,:]

        return derivative


    def gradiant_moments(self):
        '''gradiant for moment constraints '''
        return





