import numpy as np
import pandas as pd
import scipy.stats as stats

# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose: define functions help generate a random sample for logit model test
# 
#
# Definition of several variables in this file:
# n: number of individual
# c: number of products
#    outside option normalized to 0 and not included among the choices
# m: length of beta, number of variables
# ------------------------------------------------------------------------


def gen_randomsample(num_of_consumer, choice, num_of_attributes, random_seed ):
    # num_of_consumer: scalar
    # choice: tuple (c_min,c_max,tc)
    # num_of_attributes: tuple (total, for product, for consumer)

    # 0. Initialize parameters -----------------------------------------------
    # number of consumers
    n = num_of_consumer
    # range of choices, total number of products
    c_min,c_max,tc = choice
    # number of attributes
    m,m1,m2 = num_of_attributes
    # random seed
    np.random.seed(seed=13344)

    # I. id ------------------------------------------------------------------
    consumer_id = list(range(0,n))
    product_id = list(range(0,tc))

    # II. generate attributes and TC (total product) sample ------------------
    # 1. varnames
    prod_varnames = ['prod_var1']
    for i in range(1,m-1):
        varname = 'product_var' + str(i+1)
        prod_varnames.append(varname)

    # 2. set up mean and std of each attributes:
    locs = np.random.uniform(low =-50, high=50,size=m-1)
    stds = np.random.lognormal(mean = 0, sigma=5, size=m-1)

    # 3. get the value of each attributes
    p_attrs = np.random.multivariate_normal(mean = locs, cov = np.diag(stds), size = tc)
    price = np.random.lognormal(mean = 5, sigma=3, size=tc)

    # III. consumer attributes -----------------------------------------------
    # 1. var names
    consumer_varnames = ['consumer_var1']
    for i in range(1,m2):
        varname = 'consumer_var' + str(i+1)
        consumer_varnames.append(varname)

    # 2. random draw consumer attributes
    c_attrs = np.random.randint(2, size=(n, m2))

    # 3. random draw: which consumer was offered which products
    # each consumer: number of products offered
    num_prod_offered = np.random.randint(low =c_min, high=c_max+1,size=n) # c_max+1 to make the interval closed
    # each consumer: which products are offered 
    prod_offered = list(map(lambda c: np.random.choice(product_id, size=c, replace=False),
                   num_prod_offered))

    # IV. merge ---------------------------------------------------------------
    # 4. merge all attributes into a dataframe, each row is one consumer-product(offered). 
    # so the length != n*tc. the length is \sum c_{i}
    # (1) dataframe of consumers
    df_consumer = pd.DataFrame(num_prod_offered, 
                  columns = ['num_prod_offered'], 
                  index = consumer_id)
    df_consumer['prod_offered'] = prod_offered
    df_consumer.index.name = 'consumer_id'

    # (2) reshape to consumer-product(offered)
    df_consumer = df_consumer.loc[np.repeat(df_consumer.index.values, df_consumer['num_prod_offered'])]
    for consumer, frame in df_consumer.groupby(level = 0):
        prod_offered_to_thisperson = list(frame['prod_offered'])[0]
        df_consumer.loc[df_consumer.index == consumer, 'product_id'] = prod_offered_to_thisperson

    # (3) product dataframe
    df_prod = pd.DataFrame(p_attrs, 
                       index = product_id,
                       columns=prod_varnames)
    df_prod['price'] = price
    df_prod.index.name = 'product_id'

    # (4) merge
    # merge consumer id with the products assigned to each consumer
    df = pd.merge(df_consumer,
              df_prod,
              left_index = False, left_on = 'product_id', 
              right_index = True).sort_index(0)
    df = df.astype({'product_id': 'int'})

    # merge in consumer attributes
    df = pd.merge(df,
              pd.DataFrame(c_attrs, columns = consumer_varnames),
              left_index = True, 
              right_index = True).sort_index(0)
    df.index.name = 'consumer_id'
    for i in range(-m2,0):
        df[prod_varnames[i]] = df[prod_varnames[i]]*df[consumer_varnames[i]]

    # (5) clean the datafram
    df.reset_index(inplace = True)
    #df = df.set_index(['consumer_id', 'product_id'])
    return df







