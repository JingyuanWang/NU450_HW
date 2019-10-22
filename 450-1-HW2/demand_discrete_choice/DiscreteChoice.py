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
# reg:
#   k: length of beta = length of product attributes (no constant)
#   q: length of IV+exogenous RHS vars = moment conditions ------------------------------------------------------------------------
# test
'''

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools as it
import copy

# random seed
np.random.seed(seed=13344)

class DiscreteChoice:
    ''' Samples for logit demand analysis
    main part: a dataframe storing ids, Y(choices), X(attributes)'''

    # get the main dataframe (and get the set of variables)
    def __init__(self, df_consumer, consumer_ids, df_product, product_ids, true_parameters = None, df_consumer_product=None):

        # get the data
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
    # 1. construct panel
    def construct_panel_consumer_product(self, df_consumer, consumer_ids, df_product, product_ids):

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

    def simulation():
        self.simulate_consumer_choice()
        self.simulate_firm_profit()

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

        # I. make a choice
        # 1. calculate utility for each consumer-product
        self.consumer_product['utility_ijm'] = self.consumer_utility_ij(product_attribute_observed, product_attribute_unobs, price, taste, price_sensitivity, eps)
        
        # 2. get the maxium utility for each consumer 
        maxscore = self.consumer_product.groupby(['consumer_id','market_id']).agg({'utility_ijm':np.max}).rename(columns = {'utility_ijm':'max_u_im'})
        self.consumer_product = pd.merge(self.consumer_product, maxscore, 
                                        left_on = ['consumer_id','market_id'], 
                                        right_on = ['consumer_id','market_id'])

        # 3. chose the product with max utility
        self.consumer_product['choice_im'] = ( self.consumer_product['utility_ijm'] == self.consumer_product['max_u_im'] ).astype(int)

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
        total_welfare = self.consumers[['market_id','utility']].groupby('market_id').sum()

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
        return total_welfare

    # Firm:
    def firm_marginal_cost(self, cost_attribute_observed, cost_input_coeff, error_term):
        '''simulate firm's margianl cost'''
        df = self.products 
        marginal_cost = df[cost_attribute_observed].values@cost_input_coeff[1:] + cost_input_coeff[0] + df[error_term]
        return marginal_cost 

    def simulate_firm_profit(self, cost_attribute_observed, cost_input_coeff, error_term):
        '''simulate firm's profit, given sales quantity, marginal cost, and prices'''

        # 1. calculate profit for each market-product
        self.products['marginal_cost'] = self.firm_marginal_cost(cost_attribute_observed, cost_input_coeff, error_term)
        self.products['profit_jm'] = (self.products['price'] - self.products['marginal_cost']) * self.products['sales']
        
        # 2. output total profit for each firm:
        total_profit = self.products[['product_id','profit_jm']].groupby('product_id').sum().rename(columns = {'profit_jm':'total_profit'})

        return total_profit


    # III. Estimate ---------------------------------------------------
     




    # IV. visualize  ------------------------------------------------------




