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

# ---------------------------------------------------------------------------------
# function 
# ---------------------------------------------------------------------------------
def add_outside_option(df_product, product_ids):
    '''add outside options '''

    df = df_product.copy()
    market_id = product_ids['market_id']
    product_id = product_ids['product_id']

    # get a row for each market
    outside_option = df.groupby(market_id).first()
    
    # want to assign id = 0 to the outside option ==> first replace the ids for all product to id+1
    df[product_id] = df[product_id] + 1

    # replace the values to 0 
    for var in outside_option.columns:
        outside_option[var] = 0
    

    output = (df.append(outside_option.reset_index(), ignore_index=True, sort = False)
              .sort_values([market_id, product_id])
              .reset_index(drop=True)
             )

    # share should sum up to 1
    total_shares = output.groupby('market_id').agg({'shares':np.sum}).rename(columns = {'shares':'total_share'})
    output = output.merge(total_shares, on = 'market_id')
    output.loc[output['product_id'] == 0,'shares'] = 1 - output['total_share']
    output.drop(columns = 'total_share', inplace = True)
    
    return output

def random_draw(n, seed):
    '''draw n v_ips randomly from N(0,1)
    
    Input:
    --n, int, length of the output vector.
    --seed, random seeed'''
    np.random.seed(seed=seed)
    
    #v = np.random.normal(loc=0.0, scale=1.0, size=n)
    v = np.random.lognormal(mean=0.0, sigma=1.0, size=n)
    
    return v


# ---------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------

class DiscreteChoice:
    '''Samples for Discrete Choice Demand system estimation
    main part: 
    -- 3 dataframs, consumers, products, consumer_product
    -- functions: 
       -- stats
       -- simulation
       -- estimation '''

    # get the main dataframe (and get the set of variables)
    def __init__(self, df_consumer, consumer_ids, df_product, product_ids, include_outside_option, true_parameters = None, df_consumer_product=None):
        '''initialize a dataclass for discrete demand system analysis
        
        Input:
        (1) consumer
        -- df_consumer: dataframe of consumers, 1 row = 1 consumer or 1 market-consumer
        -- consumer_ids: list of str, variable names of consumer ids in df_consumer
        (2) products
        -- df_product: dataframe of products, 1 row = 1 product or 1 market-product 
        -- product_ids: list of str, variable names of consumer ids in df_product
        -- include_outside_option: True or False, True: want to add a row of 0 for each market. 
                                              False: do not need to add. if the df_product alread have a row of 0, or consumers are forced to buy something
        (3) optional input:
        -- true_parameter: if it's simulated data and have true_parameters, input a dictionary and save
        -- df_consumer_product: a merged and expanded panel df, 1 row = 1 consumer-product. (ready for MLE analysis) 
                                if input = None, will generate one automatically '''
        
        # get the data
        if include_outside_option:
            self.products = add_outside_option(df_product, product_ids)
        else:
        	self.products = df_product.sort_values([product_ids['market_id'], product_ids['product_id']]).reset_index(drop=True)
        self.consumers = df_consumer

        self.num_of_market = len(df_product[product_ids['market_id']].unique())
        self.num_of_prod = len(df_product[product_ids['product_id']].unique())

        # construct consumer-product long panel
        if df_consumer_product != None:
        	self.consumer_product = df_consumer_product
        else:
            self.consumer_product, self.panel_ids = self.construct_panel_consumer_product(self.consumers, consumer_ids, self.products, product_ids)

        # save true parameters
        if true_parameters != None:
            self.true_par = true_parameters

    # I. basic functions ----------------------------------------------------
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
        total_welfare = self.consumers.loc[self.consumers['choice'] != 0,'utility'].sum()

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
        self.products.loc[ self.products['product_id'] == 0, 'marginal_cost'] = 0 # outside option
        self.products['profit_jm'] = (self.products[price] - self.products['marginal_cost']) * self.products['sales']
        
        # 2. output total profit for each firm:
        total_profit = self.products[['product_id','profit_jm']].groupby('product_id').sum().rename(columns = {'profit_jm':'total_profit'})

        print('Total Profits of each firm (product) = ')
        print(total_profit)


    # III. Estimate --------------------------------------------------------------------

    # ------------------------------------------
    # function group 1: 
    # given delta, get shares, use build in integral
    # ------------------------------------------
    def products_social_ave_valuation_TO_market_share(self, delta, sigma_p , price, worried_about_inf = False, n_sample = 5000):
        '''A function calculate market share using given (product social value) delta, using simulated integral 
        
        Input:
        --n_sample: number of draws in the simulated integral
        '''
        
        # 0. Prepare
        # take data of interest:
        df = self.products.copy()
        df = df.sort_values(['market_id','product_id'])
        price = df[price].values[:,np.newaxis]

        # set parameter
        num_of_market = self.num_of_market
        num_of_prod = self.num_of_prod

        # construct a matrix (of 1s and 0s) with cluster diagnals 
        one_markect = np.ones((num_of_prod,num_of_prod))
        market_cluster = np.kron(np.eye(num_of_market), one_markect)

        # 1. Randomly draw a sample of random taste
        v = np.random.lognormal(0, 1, (1, n_sample) )
        v_all = np.repeat(v, num_of_market*num_of_prod, axis = 0) # to vectorize the simulation

        # 2. Calculate Each Consumer's Preference
        # --1 get the social average valuation
        delta_all = np.repeat(delta, n_sample, axis = 1)

        # --2 random taste
        random_taste = - price * v_all

        # --3 score on each product
        score_all = np.exp(random_taste) * np.exp(delta) 

        # --4 probability of chosing each product
        total_score = (market_cluster@score_all) 
        share_all = score_all / (total_score + 1)

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
            total_score = (market_cluster@score_all) 
            share_all = score_all / (total_score + outside)

        # 3. average across all simulated values
        shares = (np.sum(share_all, axis = 1) / n_sample)


        return shares






