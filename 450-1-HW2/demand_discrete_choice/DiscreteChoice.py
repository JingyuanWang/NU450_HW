'''
# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose:
# define class: Discrete Choice, with method:
#     1. choice_probability(I,X,beta)
#     2. loglikelihood(Y,I,X,beta)
#     3. ...
#
# Definition of several variables in this file:
# data:
#   i: index of consumer
#   I: total number of consumers
#   j: number of products
#   J: total number of products
# reg:
#   k: length of beta = length of product attributes (no constant)
#   q: length of IV+exogenous RHS vars = moment conditions 
# ------------------------------------------------------------------------
'''


import numpy as np
import pandas as pd
import os,sys,inspect
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import itertools as it
import copy
import importlib

# mine
directory = '/Users/jingyuanwang/GitHub/NU450_HW'
if directory not in sys.path:
    sys.path.insert(0,directory)  
import myfunctions as mf
importlib.reload(mf)

# random seed
np.random.seed(seed=13344)

class DiscreteChoice:
    '''Samples for Discrete Choice Demand system estimation
    main part: 
    -- 3 dataframs, consumers, products, consumer_product
    -- functions: 
       -- stats
       -- simulation
       -- estimation '''

    # get the main dataframe (and get the set of variables)
    def __init__(self, df_consumer, consumer_ids, df_product, product_ids, add_outside_option, true_parameters = None, df_consumer_product=None):
        '''initialize a dataclass for discrete demand system analysis
        
        Input:
        (1) consumer
        -- df_consumer: dataframe of consumers, 1 row = 1 consumer or 1 market-consumer
        -- consumer_ids: list of str, variable names of consumer ids in df_consumer
        (2) products
        -- df_product: dataframe of products, 1 row = 1 product or 1 market-product 
        -- product_ids: list of str, variable names of consumer ids in df_product
        -- add_outside_option: True or False, True: want to add a row of 0 for each market. 
                                              False: do not need to add. if the df_product alread have a row of 0, or consumers are forced to buy something
        (3) optional input:
        -- true_parameter: if it's simulated data and have true_parameters, input a dictionary and save
        -- df_consumer_product: a merged and expanded panel df, 1 row = 1 consumer-product. (ready for MLE analysis) 
                                if input = None, will generate one automatically '''
        
        # get the data
        if add_outside_option:
            self.add_outside_option(df_product, product_ids)
        self.products = df_product
        self.consumers = df_consumer
        if df_consumer_product != None:
        	self.consumer_product = df_consumer_product
        else:
            self.consumer_product, self.panel_ids = self.construct_panel_consumer_product(self.consumers, consumer_ids, self.products, product_ids)

        # save true parameters
        if true_parameters != None:
            self.true_par = true_parameters

    # I. basic functions ----------------------------------------------------
    # 0. add outside option to df_product
    def add_outside_option(self, df_product, product_ids):
        df = df_product.copy()
        # want to assign id = 0 to the outside option ==> first replace the ids for all product to id+1
        df['product_id'] = df['product_id'] + 1
        
        # for each market, add the outside option
        variables = df_product.columns
        
        for year, frame in df.groupby(['market_id']):
            # (1). get choice set for each year 
            one_row = frame.iloc[0]
            for var in variables:
                one_row[var] = 0
            frame.append(one_row)
    
    # 1. construct panel
    def construct_panel_consumer_product(self, df_consumer, consumer_ids, df_product, product_ids):
        '''construct the dataframe consumer_product (length = I*J) from cosumers dataframe (length=I) and products dataframe (lenght = J)
        
        Input:
        --df_consumer
        --consumer_ids: list of strs, names of consumer id variables
        --df_product
        --product_ids: list of strs, names of product id variables

        Note:
        ids much be continuous. This can be easily relaxed by changing the count().max() to just max() '''

        num_of_market = max(len(df_product[consumer_ids['market_id']].unique()),  
                            len(df_product[product_ids['market_id']].unique()) )
        max_num_of_consumer_withinmarket = df_consumer[consumer_ids].groupby(consumer_ids['market_id']).count().max()[0]
        max_num_of_prod_withinmarket = df_product[product_ids].groupby(product_ids['market_id']).count().max()[0]

        # construct panel ids
        panel_ids = copy.deepcopy(consumer_ids)
        panel_ids.update(product_ids)
        panel_index = pd.DataFrame(it.product(range(num_of_market), 
                                range(max_num_of_consumer_withinmarket), 
                                range(max_num_of_prod_withinmarket) ),
                                columns = panel_ids )

        # merge in product variables
        df_panel = pd.merge(panel_index, df_product, 
                 how= 'right',
                 left_on = [ panel_ids[key] for key in ('market_id','product_id')],
                 right_on =  list(product_ids.values()) )

        # merge in consumer variables
        df_panel = pd.merge(df_panel, df_consumer, 
                    how = 'right',
                    left_on = [ panel_ids[key] for key in ('market_id','consumer_id')],
                    right_on =  list(consumer_ids.values()) )

        # clean
        df_panel.sort_index(inplace = True)
        df_panel['eps_ijm'] = np.random.gumbel(0, 1, len(df_panel) )

        return df_panel, panel_ids

    # 2. set ids
    # give ids a standardized name: product_id, consumer_id, market_id


    # II. Simulate consumer choice ---------------------------------------------------
    # give all the observables, unobservables, and true parameters
    # simulate choice, estimate consumer welfare, and firm profit

    # Consumers:
    def consumer_utility_ij(self, product_attribute_observed, product_attribute_unobs, price, taste, price_sensitivity, eps):
        '''utility function (matrix operation), input are variable names, output is a vector '''
        df = self.consumer_product
        utility = df[product_attribute_observed].values@taste + df[product_attribute_unobs] + df[price_sensitivity] * df[price] + df[eps]
        
        return utility

    def simulate_consumer_choice(self, product_attribute_observed, product_attribute_unobs, price, taste, price_sensitivity, eps):
        ''' simulate consumer choice for each i, given all RHS variables including observables and unobservables.
        the input df have to be a dataframe at consumer-product level, will input self.consumer_product automatically
            Inputs:
            --product_attribute_observed: list of str, product attributes names, X in model 
            --product_attribute_unobs: str, variable name, xi in model, empirically this is back out by estimation
            --price: str, price variable name
            --taste: array, same length as product_attribute_observed,  consumer taste, beta in model.utlity function, 
            --price_sensitivity: str, variable name, consumer utility coeff of money, alpha in model 
            --eps: str, variable name
            
            Outputs:
            --utility_ij : add a column to the main datafram self.consumer_product 
            --choice_i : add a binary var column to the main dataframe, 1 means buy, 0 means not buy'''
        
        # 0. make sure there were not such variables: 
        #    consumer_product: utility_ijm, max_u_im, choice_im
        #    consumers: choice, utility
        #    products: sales
        for var in ['utility_ijm', 'max_u_im', 'choice_im']:
            if var in self.consumer_product.columns:
                self.consumer_product.rename(columns = {var: '{}_old'.format(var)}, inplace = True)
        for var in ['choice','utility']:
            if var in self.consumers.columns:
                self.consumers.rename(columns = {var: '{}_old'.format(var)}, inplace = True)
        for var in ['sales']:
            if var in self.products.columns:
                self.products.rename(columns = {var: '{}_old'.format(var)}, inplace = True)

        # I. make a choice
        # 1. calculate utility for each consumer-product
        self.consumer_product['utility_ijm'] = self.consumer_utility_ij(product_attribute_observed, product_attribute_unobs, price, taste, price_sensitivity, eps)
        
        # 2. get the maxium utility for each consumer 
        maxscore = self.consumer_product.groupby(['consumer_id','market_id']).agg({'utility_ijm':np.max}).rename(columns = {'utility_ijm':'max_u_im'})
        self.consumer_product = pd.merge(self.consumer_product, maxscore, 
                                        left_on = ['consumer_id','market_id'], 
                                        right_on = ['consumer_id','market_id'])

        # 3. chose the product with max utility
        self.consumer_product['choice_im'] = (self.consumer_product['utility_ijm'] == self.consumer_product['max_u_im'] ).astype(int)
        self.consumer_product.loc[self.consumer_product['max_u_im'] <= 0 ,'choice_im'] = 0

        # II. output
        # 1. save to consumer dataframe
        choice = (self.consumer_product
                         .loc[self.consumer_product['choice_im'] == 1 ]
                         .pivot_table(values = ['product_id', 'max_u_im'],
                                              index=['market_id', 'consumer_id'] )
                         .rename(columns = {'product_id':'choice',
                                            'max_u_im':'utility'}) )
        self.consumers = pd.merge(self.consumers, choice, 
                        how = 'left',
                        left_on =['market_id','consumer_id'],
                        right_index = True)

        # total welfare:
        total_welfare = self.consumers['utility'].sum()

        # 2. save the total sales to product dataframe
        sales = (self.consumer_product.pivot_table(values = 'choice_im',
                                      index=['market_id', 'product_id'],
                                      aggfunc={'choice_im':np.count_nonzero})
                                  .rename(columns = {'choice_im': 'sales'}) )
        self.products = pd.merge(self.products, sales, 
                how = 'left',
                left_on =['market_id','product_id'],
                right_index = True)

        # 3. return total welfare
        print('Total welfare = {}'.format(total_welfare))


    # Firm:
    def firm_marginal_cost(self, cost_attribute_observed, cost_input_coeff, error_term):
        '''simulate firm's margianl cost'''
        df = self.products 
        marginal_cost = df[cost_attribute_observed].values@cost_input_coeff[1:] + cost_input_coeff[0] + df[error_term]
        return marginal_cost 

    def simulate_firm_profit(self, cost_attribute_observed, price, cost_input_coeff, error_term):
        '''simulate firm's profit, given sales quantity, marginal cost, and prices'''

        # 0. make sure there were not such variables: 
        #    products: marginal_cost, profit_jm
        for var in ['marginal_cost', 'profit_jm']:
            if var in self.products.columns:
                self.products.rename(columns = {var: '{}_old'.format(var)}, inplace = True)

        # 1. calculate profit for each market-product
        self.products['marginal_cost'] = self.firm_marginal_cost(cost_attribute_observed, cost_input_coeff, error_term)
        self.products['profit_jm'] = (self.products[price] - self.products['marginal_cost']) * self.products['sales']
        
        # 2. output total profit for each firm:
        total_profit = self.products[['product_id','profit_jm']].groupby('product_id').sum().rename(columns = {'profit_jm':'total_profit'})

        print('Total Profits of each firm (product) = ')
        print(total_profit)


    # III. Estimate ---------------------------------------------------

    # a function: given delta, get s, constraint of 
    def products_social_ave_valuation_TO_market_share_slowbutaccurate(self, sigma_p, delta, price, print_progress = False):
        ''' A function predict the market share of each product 
        using product social average valuation, delta, 
        and parameters on consumers' tastes randomness on x, an observable product character.
        
        Input:
        --deltas is a series of product characterisctic: num_prod by 1. 
        --sigma_p is a scalar, variance of random taste on price 
        --price, str, price col name in df_input 
        --self, self gives the products dataset, including all product attributes'''

        df = self.products.copy()

        shares = np.zeros(len(df))
        
        # could write a simulation though, but since it's just 1 dim, use build-in integral function from scipyß
        for i in range(len(shares)):
            shares[i] = integrate.quad(lambda v: 
                                      self.one_consumer_prob_on_each_prod( v_ip = v, delta = delta, sigma_p = sigma_p, price = price)[i] * stats.norm.pdf(v)
                                      ,-np.inf,
                                      np.inf)[0]
            if print_progress:
                if i % 20 == 0:
                    print('complete simulating {}th element of the share'.format(i))

        return shares

    def products_social_ave_valuation_TO_market_share(self, sigma_p, delta, price, print_progress = False):
        ''' A function predict the market share of each product 
        using product social average valuation, delta, 
        and parameters on consumers' tastes randomness on x, an observable product character.
        
        Input:
        --deltas is a series of product characterisctic: num_prod by 1. 
        --sigma_p is a scalar, variance of random taste on price 
        --price, str, price col name in df_input 
        --self, self gives the products dataset, including all product attributes'''

        df = self.products.copy()

        shares = np.zeros(len(df))
        
        # could write a simulation though, but since it's just 1 dim, use build-in integral function from scipyß
        for i in range(len(shares)):
            shares[i] = integrate.quad(lambda v: 
                                      self.one_consumer_prob_on_each_prod( v_ip = v, delta = delta, sigma_p = sigma_p, price = price)[i] * stats.lognorm.pdf(v)
                                      ,0,
                                      np.inf)[0]
            if print_progress:
                if i % 20 == 0:
                    print('complete simulating {}th element of the share'.format(i))

        return shares

    def one_consumer_utility_driven_by_random_coeff(self, df, v_ip , sigma_p, price):
        '''Calculate the random utility part for 1 consumer

        Input:
        --v_ip is 1-dim 

        Note:
        This function is supposed to have another input: product_attribute_observed,
        but in the HW, only price sensitivity is random '''    
        
        mu_ij = -df[price] * sigma_p * v_ip
         
        # return a vector, length = i*j-by-1, same order as self.products
        return mu_ij

    def one_consumer_prob_on_each_prod(self, v_ip , delta, sigma_p, price):
        '''v_ip is 1-dim '''
        #df = df_input.copy()
        df = self.products.copy()
        
        # 1.calculate the random part:
        mu = self.one_consumer_utility_driven_by_random_coeff(df, v_ip, sigma_p, price)
        
        # 2.the score for each product
        df['score'] = delta + mu
        # later on we will do \[prob = exp(score)/sum( exp(score)  of all product offered) \] for each person
        # to avoid exp(something large) = inf:
        #     for each person, 
        #         devide the denominator and numerator by exp(max product score for this person)
        #         equivalent to 
        #         score for each product - max score 
        #maxscore = df.groupby('market_id').agg({'score':np.max}).rename(columns = {'score':'max_score'})
        #df = pd.merge(df,maxscore, how = 'left', left_on = 'market_id', right_on = 'market_id')
        #df['score'] = df['score'] - df['max_score']
        
        # 3. calculate a probability of chosing each product for each consumer, based on the score
        df['expscore'] = np.exp(df['score'])
        total_expscore = df.groupby('market_id').agg({'expscore': np.sum}).rename(columns = {'expscore':'total_expscore'})
        # notice here we ignore outside option for a sec
        df = pd.merge(df, total_expscore, how = 'left', left_on = 'market_id', right_on = 'market_id')
        #df['prob'] = df['expscore']/ (df['total_expscore'] + np.exp(0 - df['max_score']) ) # re-consider the outside option
        df['prob'] = df['expscore']/ (df['total_expscore'] + 1 ) # re-consider the outside option

        # 4. return 
        probability = df['prob']_
        
        return probability
    
    # IV. visualize  ------------------------------------------------------




