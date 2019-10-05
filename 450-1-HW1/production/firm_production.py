'''
# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose:
# define class: Productivity, with method:
#     1. balance()
#     2. ()
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

class firm_production:
    ''' Samples for productivity analysis
    main part: a dataframe storing firm ids, year, productions, factor inputs.'''

    # get the main dataframe (and get the set of variables)
    def __init__(self, df, firm_id, year_id, industry_id = None):
        
        # get the data
        self.full_sample = df

        # set parameters:
        self.variables = df.columns.tolist()
        self.set_ids(firm_id, year_id, industry_id)

    # I. set variables ----------------------------------------------------
    # 1. ids

    def set_ids(self, firm_id, year_id, industry_id = None):
        '''Set case_id and choice_id for the dataframe '''

        # (1) covert ids into str
        if isinstance(firm_id, list):
            if len(firm_id) == 1:
                firm_id = firm_id[0]
            else:
                raise Exception('Please input only 1 variable name')
        if isinstance(year_id, list):
            if len(year_id) == 1:
                year_id = year_id[0]
            else:
                raise Exception('Please input only 1 variable name')

        # (2) set ids
        self.firm_id = firm_id
        self.year_id = year_id
        if industry_id != None:
            self.industry_id = industry_id
        else:
            self.industry_id = '0'


        # (3) check the uniqueness:
        # a. required to be unique
        # check firm_id * year_id uniquely define the data
        dup = self.full_sample.groupby([firm_id,year_id]).size().rename('dup')
        problematic_ids = dup[dup>1].index.unique().to_numpy()
        if len(problematic_ids) != 0:
            self.problematic_ids = problematic_ids
            raise Exception('''firm_id ({}) * year_id ({}) not unique in the dataframe.
                Return: self.problematic_ids, a series of id tuples which have more than 1 obs in the dataframe'''.format(firm_id,year_id))
        
        # sort according to case-choice
        self.full_sample = self.full_sample.sort_values([firm_id, year_id]).reset_index(drop = True)
        self.full_sample.index.name = 'firm_year_id'


    # 2. get a balanced panel
    def balance_panel(self):
        '''Get a balanced panel and save as self.balancesample'''

        df = self.full_sample.set_index(self.firm_id)
        # drop the firms that change industry categories across time
        # if the data have multiple industries
        if self.industry_id != '0':
            multi_ind = (self.full_sample.groupby(self.firm_id)
                           .agg({self.industry_id: np.std})
                           .rename(columns ={self.industry_id: 'change_ind'})
                           )
            list_firms = multi_ind[multi_ind['change_ind'] > 0].index.to_list()
            df = df.drop(index = list_firms)

        # get the number of years of each obs & merge back in
        # df = self.full_sample
        length = df.groupby([self.firm_id]).size()
        df = pd.merge(df, length.rename('length'), 
                      how = 'left', 
                      left_on = self.firm_id, 
                      right_on = self.firm_id)

        # get the total length of years
        T = np.max(length)

        # 
        self.balanced_sample = (df[df['length']==T]
                                    .drop(columns='length')
                                    .reset_index()
                                    .sort_values(self.firm_id))






