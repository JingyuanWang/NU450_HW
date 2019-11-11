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
    def __init__(self, DiscreteChoice, true_parameters = None, update = False):
        '''initialize a dataclass for discrete demand system analysis
        
        Input:
        -- DiscreteChoice, classed defined in DiscreteChoice module in the same folder '''
        
        # get the data: only need the products attributes. In practise, we know nothing about the consumer.
        self.products = DiscreteChoice.products.sort_values(['market_id','product_id']) # must sorted in this way

        # parameters
        self.num_of_market = DiscreteChoice.num_of_market
        self.num_of_prod = DiscreteChoice.num_of_prod

        # prepare for MPEC estimation
        self.true_parameters = true_parameters
        self.prepare_for_MEPC()
        self.true_values( update = update)

    # I. basic functions ----------------------------------------------------

    def prepare_for_MEPC(self, n_sample = 5000):
        '''Prepare  for MEPC:
        1. save dimension: number of market M, number of product J, JM
        2. draw random sample (for simulated integral)
        3. save a matrix of 0s and 1s (to be a "cookie cutter" for some matrix operation)
        4. save a vector of market shares (to enter the constrain of MPEC)

        Input:
        -- n_sample: number of draws in the simulated integral part.
        '''
        df = self.products.sort_values(['market_id','product_id']).reset_index(drop =True)

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
        self.JM = num_of_market * num_of_prod

        # specify variable name
        self.varname_randomcoeff = 'price'

        return


    def true_values(self, update = True):
        ''' If true delta is given, check whether the true delta and the true share matches.
        If not, replace the true share with simulated share (function of true delta).'''

        df = self.products.sort_values(['market_id','product_id']).reset_index(drop = True)
        true_parameters = self.true_parameters
        
        # True variables
        # 1. Delta
        product_attribute_observed = ['x1','x2','x3']
        product_attribute_unobs = 'xi'
        price = 'price'

        delta_true = (df['xi'] + df[product_attribute_observed].values@true_parameters['beta'] \
              - df[price]*true_parameters['alpha_0']).values

        delta_true = delta_true[:, np.newaxis]

        # 2. Share
        if update:
            price = self.varname_randomcoeff
            share_true = self.products_social_ave_valuation_TO_market_share(delta = delta_true, sigma_p=1 , price_varname = price)
        else:
            share_true = self.MPEC_par_true_share = df['shares'].values[:,np.newaxis]
        
        # save
        self.MPEC_par_true_delta = delta_true
        self.MPEC_par_true_share = share_true
        return 

    # III. Generate Variables --------------------------------------------------------------------

    # ------------------------------------------
    # function group 1: 
    # generate exogenous variables
    # ------------------------------------------
    def construct_exogenous_var(self, first_stage_check = False, exogenous_varname = None, endogenous_varname = None):
        '''Claim exogenous and endogenous varaibles
        Construct all potential IVs, do first stage test

        Note: 
        1. all input must be list (could be list of 1 element)
        2. exogenous_varname and endogenous_varname can not be NONE if first_stage_check = True'''

        # save exogenous var names
        #self.exogenous_var = exogenous_varname

        # generate IV
        self.gen_BLP_instruments()
        self.gen_Hausman_instrument()

        # check first stage for the IVs
        if first_stage_check:
            # check BLP instrument
            var = self.exogenous_var_BLPinstruments 
            self.first_stage(exogenous_varname = exogenous_varname, endogenous_varname = endogenous_varname, IV_varname = var)
            
            # check hausman instrument
            var = self.exogenous_var_Hausmaninstruments
            self.first_stage(exogenous_varname = exogenous_varname, endogenous_varname = endogenous_varname, IV_varname = var)

            # check both
            # (1) all variables
            var = self.exogenous_var_BLPinstruments + self.exogenous_var_Hausmaninstruments
            self.first_stage(exogenous_varname = exogenous_varname, endogenous_varname = endogenous_varname, IV_varname = var)

            # (2) just 1 hausman instruments
            var = self.exogenous_var_BLPinstruments + [self.exogenous_var_Hausmaninstruments[0]]
            self.first_stage(exogenous_varname = exogenous_varname, endogenous_varname = endogenous_varname, IV_varname = var)

        return
    
    def gen_BLP_instruments(self):
        ''' generate hauseman instrument
        Output:
        -- save the variable in self.products data frame
        -- save the varaible names in self.exogenous_var_BLPinstruments '''

        output_varnames = ['x2_other1', 'x2_other2' , 'x3_other1' , 'x3_other2']

        df = self.products.copy()
        df.drop(columns = output_varnames, errors = 'ignore', inplace = True)

        func1 = lambda series: pd.DataFrame({'rolling': np.roll(series.values, 1).tolist() })
        func2 = lambda series: pd.DataFrame({'rolling': np.roll(series.values, 2).tolist() })    

        df['x2_other1'] = df.groupby('market_id')['x2'].apply(func1).reset_index(drop = True)
        df['x2_other2'] = df.groupby('market_id')['x2'].apply(func2).reset_index(drop = True)
        df['x3_other1'] = df.groupby('market_id')['x3'].apply(func1).reset_index(drop = True)
        df['x3_other2'] = df.groupby('market_id')['x3'].apply(func2).reset_index(drop = True)

        # save
        self.products = df.sort_values(['market_id','product_id']).reset_index(drop = True)
        self.exogenous_var_BLPinstruments = output_varnames

        return 

    def gen_Hausman_instrument(self):
        ''' generate hauseman instrument
        Output:
        -- save the variable in self.products data frame
        -- save the varaible names in self.exogenous_var_Hausmaninstruments '''
        output_varnames = ['price_others', 'price_others_sq']

        df = self.products.copy()
        df.drop(columns = output_varnames, errors = 'ignore', inplace = True)
        num_of_market = self.num_of_market

        p_sum = df.groupby('product_id').agg({'price': np.sum }).rename(columns = {'price': 'price_others'} )
        df = pd.merge(df, p_sum, left_on = 'product_id', right_on = 'product_id')

        df['price_others'] = (df['price_others'] - df['price']) / (num_of_market-1)
        df['price_others_sq'] = df['price_others']**2

        # save
        self.products = df.sort_values(['market_id','product_id']).reset_index(drop = True)
        self.exogenous_var_Hausmaninstruments = output_varnames

    def first_stage(self, exogenous_varname, endogenous_varname, IV_varname):
        '''Check the first stage for IVs 
        All input must be lists '''

        import econtools 
        import econtools.metrics as mt

        # get data
        df = self.products.sort_values(['market_id','product_id']).copy()
        x_varnames = exogenous_varname + IV_varname
        
        # first stage reg
        # 1. partial out x:
        #for y_varname in endogenous_varname:
        #    result = mt.reg(df, y_varname, exogenous_varname)
        #    df['{}_ddot'.format(y_varname)] = df[y_varname] - df[exogenous_varname].values@result.beta

        #IV_varname_ddot = []
        #for iv_varname in IV_varname:
        #    result = mt.reg(df, iv_varname, exogenous_varname)
        #    df['{}_ddot'.format(iv_varname)] = df[iv_varname] - df[exogenous_varname].values@result.beta
        #    IV_varname_ddot = IV_varname_ddot + ['{}_ddot'.format(iv_varname)]

        # 2. First stage regression
        for y_varname in endogenous_varname:

            #result = mt.reg(df, '{}_ddot'.format(y_varname), IV_varname_ddot)
            result = mt.reg(df, y_varname, x_varnames)
            joint_F = result.Ftest(IV_varname) 

            print('#=======================================================')
            print('Endogenous var: {} '.format(y_varname))
            print('            IV: {} '.format(IV_varname))
            print(' ')
            print('Ftest = {}'.format(joint_F))
            print(' ')
            print('regression:')
            print(result)
            print(' ')

        return


    # III. Estimate --------------------------------------------------------------------

    # ------------------------------------------
    # function group 0: 
    # 1. claim regressers
    # 2. initialize coefficients
    # ------------------------------------------

    def MPEC_claim_var(self, independent_var, exogenous_var):
        '''independent var includes all the exogenous and endogenous variables of interests (the names)
        exogenous var in cludes all exogenous variable names, x and IV'''
        
        df = self.products.sort_values(['market_id','product_id']).reset_index(drop =True)

        self.MPEC_X = df[independent_var].values
        self.MPEC_Z = df[exogenous_var].values
        self.MPEC_W = np.eye(self.MPEC_Z.shape[1])

        # update self.JM: in case there are missings, so JM != J * M
        # save q: number of moment conditions
        (self.JM, self.q) = self.MPEC_Z.shape 

        # length of MPEC regression coefficients
        self.MPEC_parameter_length = self.JM + self.q + 1

        return 

    def initial_parameters(self, sigma_p = 1, true_par=True, maxiter = 100):
        '''set initial parameters: by default, input true value as initial value '''

        delta = self.MPEC_par_true_delta

        if true_par:        
            sigma = np.array(1,ndmin=2)
        else:
            sigma = sigma_p
            delta_init = delta
            delta = self.BACK_OUT_products_social_ave_valuation_FROM_market_share(sigma, delta_init, maxiter = maxiter)

        eta = self.initial_eta(delta)

        parameters = np.vstack( (delta, sigma, eta)  )
        return parameters

    def initial_eta(self, delta):
        '''Initialize eta: make sure it satisfies the constraint 

        Note: differences in MPEC_claim_var() will cause differences in this initial value, both the length and the value'''

        JM = self.JM
        X = self.MPEC_X
        Z = self.MPEC_Z
        W = self.MPEC_W  

        Pz = Z@np.linalg.inv(Z.T@Z)@Z.T
        Mxz = np.eye(JM) - X@np.linalg.inv(X.T@Pz@X)@X.T@Pz
        
        g = Z.T@(Mxz@delta)

        return g

    def get_alpha_beta(self, delta):
        '''Get alpha and beta using IV regressions
        (after claiming exogenous and endogenous varaibles) '''

        X = self.MPEC_X
        Z = self.MPEC_Z
        W = self.MPEC_W  

        Pz = Z@np.linalg.inv(Z.T@Z)@Z.T

        beta = np.linalg.inv(X.T@Pz@X)@X.T@Pz@delta

        return beta

    
    # ------------------------------------------
    # function group 1: 
    # 1. given delta, get shares, use build in integral
    # 2. given shares, back out delta: contraction mapping using 1
    # ------------------------------------------
    def products_social_ave_valuation_TO_market_share(self, delta, sigma_p , price_varname, for_gradient = False, worried_about_inf = False):
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
        if len(delta.shape) == 1:
            delta = delta[:,np.newaxis]
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

        if for_gradient:
            # return a matrix of num_of_market*num_of_prod by number_of_consumers, about 300 -by- 5000
            return {'share_all': share_all, 'price' : price}
        else:
            # 3. Integral: average across all simulated values
            shares = (np.sum(share_all, axis = 1) / n_sample)
            shares = shares[:,np.newaxis]
            return shares


    def BACK_OUT_products_social_ave_valuation_FROM_market_share(self, sigma_p, delta_init, maxiter=10000, tol = 1e-1):
    
        delta = delta_init
        share_true = self.MPEC_par_true_share
        price = self.varname_randomcoeff

        error = 99
        iteration = 0
        while error > tol and iteration < maxiter:
            s = self.products_social_ave_valuation_TO_market_share(delta = delta, sigma_p = sigma_p, price_varname = price)
            delta_new = delta + np.log(share_true) - np.log(s)
            
            # update
            error = np.asscalar(sum(abs(delta_new - delta_init)))
            delta = delta_new
            iteration = iteration + 1

            if iteration%500 == 0:
                print('{}: error = {}'.format(iteration, error))
        
        if error <= tol:
            print('converges in {} iterations, with tolerance = {}'.format(iteration, error))
        else:
            print('does not converge in {} iterations, current error = {}'.format(iteration, error))
        return delta

    # ------------------------------------------
    # function group 2: 
    # objective functions and constrains
    # ------------------------------------------
    
    def MPEC_obj(self, parameter):
        '''  '''
        
        JM = self.JM
        q =self.q

        if len(parameter) != (JM + q + 1) :
            raise Exception('len(parameter) != (JM + q + 1)')

        delta = parameter[0:JM]  # JM*1
        sigma = parameter[JM]     # scalar
        eta = parameter[-q:]     # k*1
        
        W = self.MPEC_W
        obj = eta.T@W@eta
        
        return obj

    
    def MPEC_constraint_share(self, parameter):

        shares = self.MPEC_par_true_share
        price_varname = self.varname_randomcoeff

        JM = self.JM
        q =self.q

        delta = parameter[0:JM]  # JM*1
        sigma = parameter[JM]     # scalar
        eta = parameter[-q:]     # k*1
        
        shares_hat = self.products_social_ave_valuation_TO_market_share(delta = delta, sigma_p=sigma, price_varname = price_varname, 
                                                                                   worried_about_inf = False)
        return  shares_hat - shares

    def MPEC_constraint_moment_conditions(self, parameter):
        
        JM = self.JM
        q = self.q
        X = self.MPEC_X
        Z = self.MPEC_Z
        W = self.MPEC_W    

        delta = parameter[0:JM]  # JM*1
        sigma = parameter[JM]     # scalar
        eta = parameter[-q:]     # k*1
        
        if Z.shape[1] == X.shape[1]:
            raise Exception('dim(Z) < # of par, not identified')
        else:
            Pz = Z@np.linalg.inv(Z.T@Z)@Z.T
            Mxz = np.eye(JM) - X@np.linalg.inv(X.T@Pz@X)@X.T@Pz
        
        g = Z.T@(Mxz@delta)
        
        return (g - eta)[:, np.newaxis]


    # ------------------------------------------
    # function group 3: 
    # gradient for objective function
    # ------------------------------------------
    def MPEC_gradient_obj(self, parameter):

        JM = self.JM
        q =self.q
        W = self.MPEC_W
        Z = self.MPEC_Z

        gradient_obj_on_delta_sigma = np.zeros( (JM+1,1) )

        gradient_obj_on_eta = self.gradient_obj(parameter, W, Z)

        if len(gradient_obj_on_eta.shape) == 1:
            gradient_obj_on_eta = gradient_obj_on_eta[:,np.newaxis]

        gradient = np.vstack( (gradient_obj_on_delta_sigma, gradient_obj_on_eta) )

        # 307-by-1
        return gradient

    def gradient_obj(self, parameter, W, Z):

        JM = self.JM
        q =self.q
        
        delta = parameter[0:JM]  # JM*1
        sigma = parameter[JM]     # scalar
        eta = parameter[-q:]     # k*1

        return 2 * W@eta

    # ------------------------------------------
    # function group 4: 
    # gradient for constraint
    # ------------------------------------------

    def MPEC_gradient_constraints(self, parameter):

        JM = self.JM
        q =self.q
        X = self.MPEC_X
        W = self.MPEC_W
        Z = self.MPEC_Z
        price_varname  = self.varname_randomcoeff

        cluster_11_share_on_delta_sigma = self.gradient_share(parameter, price_varname, Z)
        cluster_121_mc_on_delta = self.gradient_moment_conditions(parameter, X, Z, W)
        cluster_122_mc_on_sigma = np.zeros( (1,q))

        cluster_21_share_on_eta =  np.zeros( (q,JM) )
        cluster_22_mc_on_eta = -np.diag(np.ones(q))

        # stack
        cluster_12_mc_on_delta_sigma = np.vstack( (cluster_121_mc_on_delta, cluster_122_mc_on_sigma) )
        cluster_1_on_delta_sigma = np.hstack( (cluster_11_share_on_delta_sigma, cluster_12_mc_on_delta_sigma) )
        cluster_2_on_eta = np.hstack( (cluster_21_share_on_eta, cluster_22_mc_on_eta) )
        gradient = np.vstack( (cluster_1_on_delta_sigma, cluster_2_on_eta) )

        return gradient

    def gradient_share(self, parameter, price_varname, Z):
        '''derivatives of the share function (a set of constraints, num_of_prod * num_of_market, denoted JM) 

        Output:
        -- derivatives, (JM+1) -by- JM. 
                        each column is the derivative of one constraint on 
                               -- JM deltas (social average valuation on each product-market);
                               -- 1 sigma (taste randomness)
        '''
        (JM,q) = Z.shape

        delta = parameter[0:JM]   # JM*1
        sigma = parameter[JM]     # scalar
        eta = parameter[-q:]      # k*1

        interim_integral_results = self.products_social_ave_valuation_TO_market_share(delta, sigma , price_varname, for_gradient = True)
        share_all = interim_integral_results['share_all']
        price     = interim_integral_results['price']

        derivatives_firstJMrows = self.gradient_share_on_social_ave_valuation(share_all)

        derivatives_laterrows = self.gradient_share_on_taste_randomness(share_all,price)
        
        gradient = np.vstack((derivatives_firstJMrows, derivatives_laterrows))

        return gradient

    def gradient_share_on_social_ave_valuation(self, share_all):
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

    def gradient_share_on_taste_randomness(self, share_all, price):
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
        derivative_numerator_all =  - share_all * price * v_all

        # derivative of the denominator
        partial_all = share_all * price * v_all
        partial_sum_all = (market_cluster@partial_all) 
        derivative_denominator_all = share_all * partial_sum_all

        # derivative
        derivative_all = derivative_numerator_all + derivative_denominator_all
        derivative = (np.sum(derivative_all, axis = 1) / n_sample)
        
        # each row stands for 1 parameter; each columns stands for 1 constraint.
        derivative = derivative[np.newaxis,:]

        return derivative

    def gradient_moment_conditions(self, parameter, X, Z, W):
        '''gradient for moment constraints '''
        

        (JM,q) = Z.shape    

        delta = parameter[0:JM]  # JM*1
        sigma = parameter[JM]     # scalar
        eta = parameter[-q:]     # k*1
        
        Pz = Z@np.linalg.inv(Z.T@Z)@Z.T
        Mxz = np.eye(JM) - X@np.linalg.inv(X.T@Pz@X)@X.T@Pz

        derivative = Mxz.T@Z

        return derivative

    # ------------------------------------------
    # function group 5: 
    # Optimal Weighting matrix
    # ------------------------------------------
    # emm ... 

    # IV. Post-estimation: elasticity/...
    def elasticity():
        return 





