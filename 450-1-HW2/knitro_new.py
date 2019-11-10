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

rootpath = '/home/jwr2838/NU450_HW/450-1-HW2'
datapath = rootpath + '/' + 'data'
resultpath = rootpath + '/' + 'results'

# model parameters (true value)
true_parameters = {'beta': np.array([5,1,1]),
                   'alpha_0': 1,
                   'alpha_sigma':1,
                   'gamma': np.array([2,1,1])}

# import file names
matfiles = ['10markets3products', '100markets3products','100markets5products']
names = ['m10_prod3', 'm100_prod3', 'm100_prod5']



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



BLP_MPEC = estimation.BLP_MPEC(demand_m100_prod3, true_parameters)

BLP_MPEC.construct_exogenous_var(exogeneous_varname = ['x1', 'x2', 'x3'], endogenous_varname = ['price'], first_stage_check = False)

independent_var = ['x1', 'x2', 'x3', 'price']
exogenous_var = ['x1', 'x2', 'x3'] + [ BLP_MPEC.exogeneous_var_Hausmaninstruments[0] ] + BLP_MPEC.exogeneous_var_BLPinstruments
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

m = 308    # number of constraints
n = 309    # number of parameters

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

xInit = BLP_MPEC.initial_parameters().flatten()

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
        print("Knitro successful, feasibility violation    = {}".format(
            ks.feas_error))
        print("                   KKT optimality violation = {}".format(
            ks.opt_error))

# save the solution to x
    output = pd.DataFrame(res.x, columns = ['delta_sigma_eta'])
    outputpath = resultpath + '/' + 'Q2'
    filename  = 'result_from_knitro_opt'
    outputfile = outputpath + '/' + filename + '.csv'

    output.to_csv(outputfile)





