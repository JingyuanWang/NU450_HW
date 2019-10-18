'''
# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose:
# define class: LogitDemand, with method:
#     1. choice_probability(I,X,beta)
#     2. loglikelihood(Y,I,X,beta)
#     3. ...
#
# Definition of several variables in this file:
# n: number of cases 
# c: number of available products (choices)
#    outside option normalized to 0 and not included among the choices
# m: length of beta, number of variables/attributes
# ------------------------------------------------------------------------
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
        self.main = df

        # set parameters:
        self.columns = df.columns.tolist()
        self.set_ids(case_id, choice_id, case_groupid, choice_groupid)

        # print out information:
        print('SAMPLE:')
        print('    female only: {}; under 65 only: {}'.format(female_only == 1,under65 == 1))
        print('number of observations: {}'.format(len(df)))
        print('       number of cases: {}'.format(len(df[case_id].unique())))
        print('            choice set: {}'.format(df[choice_id].unique().tolist()))

    # I. set variables ----------------------------------------------------
    # 1. ids



    # II. Firm data --------------------


    # III. Consumer data ---------------------------------------------------

    # IV. visualize  ------------------------------------------------------
 



