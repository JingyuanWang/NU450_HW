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

class DiscreteChoice:
    ''' Samples for logit demand analysis
    main part: a dataframe storing ids, Y(choices), X(attributes)'''

    # get the main dataframe (and get the set of variables)
    def __init__(self, df, case_id, choice_id, case_groupid = None, choice_groupid = None):

        # get the data
        self.products = df_product
        self.consumers = df_consumer
        if df_consumers_products != None:
        	self.consumers_products = df_consumers_products

        # set parameters:
        self.columns = df.columns.tolist()
        self.set_ids(case_id, choice_id, case_groupid, choice_groupid)

    # I. set variables ----------------------------------------------------
    # 1. ids



    # II. Firm data --------------------


    # III. Consumer data ---------------------------------------------------

    # IV. visualize  ------------------------------------------------------
 



