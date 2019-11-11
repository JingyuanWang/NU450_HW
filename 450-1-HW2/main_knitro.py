from knitroWrapper import KnitroSolver
import numpy as np


import numpy as np
import pandas as pd
import os,sys,inspect
import scipy.stats as stats
import scipy.optimize as opt
import scipy.integrate as integrate
from scipy.io import loadmat
#import econtools 
#import econtools.metrics as mt
#import statsmodels.discrete.discrete_model as sm
#import matplotlib.pyplot as plt
import itertools as it
import copy

# mine
# (1)
import initialize
import estimation 

# (2)
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
#import myfunctions as mf

import importlib
importlib.reload(initialize)
importlib.reload(estimation)
#importlib.reload(mf)


# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose: HW2 
# 
# Import data and stats
# 
#
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# 0. set path
# ------------------------------------------------------------------------

rootpath = '/home/jwr2838/NU450_HW/450-1-HW2'
datapath = rootpath + '/' + 'data'
resultpath = rootpath + '/' + 'results'


# ------------------------------------------------------------------------
# 0. input parameters 
# ------------------------------------------------------------------------
# model parameters (true value)
true_parameters = {'beta': np.array([5,1,1]),
                   'alpha_0': 1,
                   'alpha_sigma':1,
                   'gamma': np.array([2,1,1])}

# import file names
matfiles = ['10markets3products', '100markets3products','100markets5products']
names = ['m10_prod3', 'm100_prod3', 'm100_prod5']

# load config
def loadConfig(config_filename):
    # read configs that defines the subsample

    # want info: female only? under 65 only?
    info_want = ['specification','IV_varnames','update_true_share']
    
    # import file
    cfgfile = pd.read_csv(config_filename, header=0)
    #return cfgfile
    # get info
    if not all(elem in cfgfile.columns  for elem in info_want):
        raise Exception('not all information needed are in the config file.')

    specification = cfgfile['specification'].iloc[0]
    IV_varnames = cfgfile['IV_varnames'].tolist()
    update_true_share = cfgfile['update_true_share'].iloc[0]
    return {'specification':specification, 
            'IV_varnames':IV_varnames, 
            'update_true_share':update_true_share}

config_filename = 'cfg/config.csv'
args = loadConfig(config_filename)
specification = args['specification']
IV_varnames = args['IV_varnames']
update_true_share = args['update_true_share']

# ------------------------------------------------------------------------
# I. read files 
# ------------------------------------------------------------------------
# 1 import
inputfiles = dict(zip(names, matfiles))
data = {}
for name, filename in inputfiles.items():
    file = datapath + '/Simulation Data/'+ filename
    data[name] = loadmat(file)


# 2. clean and save to a class
def save_input_to_DiscreteChoiceClass(num_of_market, num_of_prod, include_outside_option, true_parameters):
    
    sample = 'm{}_prod{}'.format(num_of_market,num_of_prod)
    df_product = (pd.DataFrame(np.concatenate( 
                              (data[sample]['x1'],
                               data[sample]['xi_all'],
                               data[sample]['w'], 
                               data[sample]['Z'],
                               data[sample]['eta']), axis = 1 ))
                  .rename(columns = {0:'x1',1:'x2',2:'x3',3:'xi', 4:'w', 5:'z',6:'eta'}) )
    df_product['price'] = np.reshape(data[sample]['P_opt'].T, (num_of_market*num_of_prod,1))
    df_product['shares'] = np.reshape(data[sample]['shares'].T, (num_of_market*num_of_prod,1))

    df_product['product_id'] = (np.repeat(np.arange(num_of_prod)[:,np.newaxis],num_of_market, axis = 1)
                                .reshape((num_of_prod*num_of_market,1), order = 'F') )

    df_product['market_id'] = np.repeat(np.arange(num_of_market)[:, np.newaxis], num_of_prod, axis=0)
    product_ids = {'market_id':'market_id',
                   'product_id':'product_id'}

    # consumer
    df_consumer = pd.DataFrame(data[sample]['alphas'].T, columns=[idx for idx in range(num_of_market) ])
    df_consumer = df_consumer.stack().reset_index().rename(columns = {'level_0':'consumer_id', 
                                                                      'level_1':'market_id',
                                                                      0:'alpha'})


    consumer_ids = {'market_id':'market_id',
                   'consumer_id':'consumer_id'}
    
    output = initialize.DiscreteChoice(df_consumer, consumer_ids, 
                                     df_product, product_ids, include_outside_option = include_outside_option,
                                     true_parameters=true_parameters)
    return output


demand_m10_prod3 = save_input_to_DiscreteChoiceClass(num_of_market=10, 
                                                     num_of_prod=3, 
                                                     include_outside_option = False,
                                                     true_parameters = true_parameters)

demand_m100_prod3 = save_input_to_DiscreteChoiceClass(num_of_market=100, 
                                                     num_of_prod=3, 
                                                     include_outside_option = False,
                                                     true_parameters = true_parameters)

demand_m100_prod5 = save_input_to_DiscreteChoiceClass(num_of_market=100, 
                                                     num_of_prod=5, 
                                                     include_outside_option = False,
                                                     true_parameters = true_parameters)


# ------------------------------------------------------------------------
# III. save the file
# ------------------------------------------------------------------------
BLP_MPEC = estimation.BLP_MPEC(demand_m100_prod3, true_parameters, update_true_share)

BLP_MPEC.construct_exogenous_var()

independent_var = ['x1', 'x2', 'x3', 'price']
exogenous_var = ['x1', 'x2', 'x3'] + IV_varnames
BLP_MPEC.MPEC_claim_var(independent_var,exogenous_var)


"""
This script is a Wrapper-Equivalent of the
example HS15Numpy from the knitro package.

Modified by Ben Vatter 2017


All relevant copyright and IP is theirs
# *******************************************************
# * Copyright (c) 2015 by Artelys                       *
# * All Rights Reserved                                 *
# *******************************************************

The description of the problem is:

#  Solve test problem HS15 from the Hock & Schittkowski collection.
#
#   min   100 (x2 - x1^2)^2 + (1 - x1)^2
#   s.t.  x1 x2 >= 1
#         x1 + x2^2 >= 0
#         x1 <= 0.5
#
#   The standard start point (-2, 1) usually converges to the standard
#   minimum at (0.5, 2.0), with final objective = 306.5.
#   Sometimes the solver converges to another local minimum
#   at (-0.79212, -1.26243), with final objective = 360.4.
# #
"""
KTR_INFBOUND = KnitroSolver.KTR_INFBOUND
KTR_CONTYPE_GENERAL = KnitroSolver.KTR_CONTYPE_GENERAL

m = BLP_MPEC.MPEC_parameter_length - 1    # number of constraints
n = BLP_MPEC.MPEC_parameter_length        # number of parameters

init_options = {
# https://www.artelys.com/docs/knitro//3_referenceManual/knitroPythonReference.html
    'objGoal': KnitroSolver.KTR_OBJGOAL_MINIMIZE,  # MIN OR MAX
    'objType': KnitroSolver.KTR_OBJTYPE_QUADRATIC, # LINEAR OR QUADRATIC OR GENERAL
    'xLoBnds': np.ones(n)*(-KTR_INFBOUND),         # PARAMETER SPACE
    'xUpBnds': np.ones(n)*KTR_INFBOUND,
    'cType': np.ones(m, np.int64) * KTR_CONTYPE_GENERAL, # CONSTRAINT TYPES
    #'cEqBnds': np.zeros(m),
    'cLoBnds': np.ones(m)*(-1e-9),
    'cUpBnds': np.ones(m)*1e-9,
    'jacIndexVars': np.mod(np.arange(m*n), n), 
    'jacIndexCons': np.repeat(np.arange(m), n),
    #'hessIndexRows': np.array([0, 0, 1], np.int32),
    #'hessIndexCols': np.array([0, 1, 1], np.int32)
}

#xInit = BLP_MPEC.initial_parameters(sigma_p = 1.02, true_par=False, maxiter = 100).flatten()
xInit = np.ones( BLP_MPEC.MPEC_parameter_length )

ktr_options = {
    'outlev': 'all',
    #'gradopt': 1,
    #'derivcheck': 1,
    'hessopt': 2,
    'feastol': 1.0e-10
}




# ******************************************************
# *                                                    *
# *  Some Class definition and initialization here     *
# *                                                    *
# *  MyClass = Class(args)                             *
# *                                                    *
# ******************************************************



def evaluateFC(x, c):

    obj = BLP_MPEC.MPEC_obj(x)
    c1 = BLP_MPEC.MPEC_constraint_share(x)
    c2 = BLP_MPEC.MPEC_constraint_moment_conditions(x)
    c0 = np.vstack( (c1,c2) ).flatten()

    #c = c0
    c[:] = c0
    #for i in range(len(c)):
    #    c[i] = c0[i]


    return obj


def evaluateGA(x, objGrad, jac):
    #print(BLP_MPEC.MPEC_gradient_obj(x))
    objGrad0 = BLP_MPEC.MPEC_gradient_obj(x).flatten()

    jac0 = BLP_MPEC.MPEC_gradient_constraints(x).T.flatten()


    objGrad[:] = objGrad0
    jac[:] = jac0
    return


    for i in range(len(objGrad)):
        objGrad[i] = objGrad0[i]

    for i in range(len(jac)):
        jac[i] = jac0[i]

    return




if __name__ == "__main__":
    with KnitroSolver(init_options, ktr_options) as ks:
        ks.objective = evaluateFC
        ks.gradient = evaluateGA
        res = ks.minimize(xInit)
        print("specification: {}".format(specification))
        print("IVs : {}".format(IV_varnames))
        print("Knitro successful, feasibility violation    = {}".format(
            ks.feas_error))
        print("                   KKT optimality violation = {}".format(
            ks.opt_error))

    # get alpha and beta
    delta_hat = res.x[:300]
    betas = BLP_MPEC.get_alpha_beta(delta_hat)

    # save the solution to x
    output_dict={'delta_sigma_eta':res.x, 
                 'betas':betas,
                 'varnames':independent_var,
                 'specification':specification,
                 'IV_varnames':IV_varnames}
    output = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in output_dict.items() ]))

    outputpath = resultpath + '/' + 'Q2'
    filename  = 'result_from_knitro_opt_{}'.format(specification)
    outputfile = outputpath + '/' + filename + '.csv'

    print(outputfile)
    output.to_csv(outputfile)





