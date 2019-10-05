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

    # I. basic settings, if input a dataframe ----------------------------
    def gen_randomsample(num_of_firm, num_of_industry = 1, year_span = (1990,1999), random_seed=13344 ):
        '''generate a random sample of unbalanced panel, mimic firms exist and entry
        input varaiables: 
            -num_of_firm: scaler
            -num_of_industry: scaler, default = 1
            -year_span: tuple, (start,end), default = (1990,1999)
            -random_seed: productivities are drawn from a Normal distribution.'''

        # 0. Initialize parameters -----------------------------------------------
        # number of consumers
        n = num_of_consumer
        # range of choices, total number of products
        c_min,c_max,tc = choice
        # number of attributes
        m,m1,m2 = num_of_attributes
        # random seed
        np.random.seed(seed=random_seed)



    # II. basic settings, if input a dataframe ---------------------------
    # 1. ids
    def set_ids(self, firm_id, year_id, industry_id = None):
        '''Set firm_id and year_id for the dataframe 
        Print basic stats about the ids'''

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
            self.industry_id = 'industry_id'


        # (3) check the uniqueness:
        # a. required to be unique
        # check firm_id * year_id uniquely define the data
        dup = self.full_sample.groupby([firm_id,year_id]).size().rename('dup')
        problematic_ids = dup[dup>1].index.unique().to_numpy()
        if len(problematic_ids) != 0:
            self.problematic_ids = problematic_ids
            raise Exception('''firm_id ({}) * year_id ({}) not unique in the dataframe.
                Return: self.problematic_ids, a series of id tuples which have more than 1 obs in the dataframe'''.format(firm_id,year_id))
        
        # (4) output
        # sort according to case-choice
        self.full_sample = self.full_sample.sort_values([firm_id, year_id]).reset_index(drop = True)
        self.full_sample.index.name = 'firm_year_id'
        self.full_sample= self.full_sample.astype({self.industry_id: 'int32',
                                                   self.year_id:'int32'})
        if industry_id == None:
            self.full_sample['industry_id'] = 0

        # (5) stats
        table = self.stats_panel_basic(df = self.full_sample)
        print("-- Table: Number of firms in each industry-year (full sample) ----")
        print(table)

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
         # (5) stats
        table = self.stats_panel_basic(df = self.balanced_sample)
        print("-- Table: Number of firms in each industry-year (balanced sample) ----")
        print(table)

    # III. statistics ----------------------------------------------------
    def stats_panel_basic(self, df, industry=None):
        '''Report basic facts about the panel data:
            -number of years
            -number of firms
            -on average/min/max, how many years a firm would survive.
        if industry_id is given, then report the facts of that industry
        else, report the full sample.'''

        pd.options.display.float_format = '{:,.0f}'.format
        if industry == None:
            table = (df.pivot_table(values=[self.firm_id], 
                        index=[self.industry_id, self.year_id], 
                        aggfunc={self.firm_id:np.count_nonzero} )
                    .rename(columns = {self.firm_id: 'Number of firms'} )
                    .reset_index().pivot(index=self.year_id, columns=self.industry_id, values='Number of firms')
                    )
            table.loc["Total"] = df.groupby(['industry_id'])['firm_id'].nunique()
        else:
            table = (df[df[self.industry_id] == industry]
                        .pivot_table(values=[self.firm_id], 
                            index=[self.industry_id, self.year_id], 
                            aggfunc={self.firm_id:np.count_nonzero}) 
                        .rename(columns = {self.firm_id: 'Number of firms'} )
                        )
            table.loc[(industry,"Total"),'Number of firms'] = df.loc[df[self.industry_id]==industry,'firm_id'].nunique()
        return table






