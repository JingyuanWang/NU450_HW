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
# To do:
# GNR jacobian
# ------------------------------------------------------------------------
'''

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt
import econtools 
import econtools.metrics as mt
import matplotlib.pyplot as plt
import statsmodels.discrete.discrete_model as sm
import importlib

# mine
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import myfunctions as mf
importlib.reload(mf)



class firm_production:
    ''' Samples for productivity analysis
    main part: a dataframe storing firm ids, year, productions, factor inputs.'''

    # get the main dataframe (and get the set of variables)
    def __init__(self, df, firm_id, year_id, resultpath, industry_id = None ):
        
        # get the data
        self.full_sample = df

        # set parameters:
        self.variables = df.columns.tolist()
        self.set_ids(firm_id, year_id, industry_id)

        # set path
        self.set_path(resultpath)


    # I. basic settings --------------------------------------------------
    # 0. save temporary files and outputs
    def set_path(self, resultpath):
        self.resultpath = resultpath

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
    # II. statistics ----------------------------------------------------
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


    # III. regressions ----------------------------------------------------

    def GNR_stage1_obj(self,beta, df, share, poly):
        x = df[poly].values@beta
        RHS = np.log(x)
        LHS = df[share].values
        return sum((LHS - RHS)**2)

    def GNR_stage1_constraint(self,df_input,poly,beta):
        tol = 1e-2
        min_value = np.min(df_input[poly].values@beta) - tol
        return df_input[poly].values@beta - tol

    def GNR_stage1_jac(self, beta, df, share, xvar):
        s =  df[share].values
        common = (s - np.log(df[xvar].values@beta)) * 2 / (df[xvar].values@beta)
        der =  - df[xvar].values * common.flatten()[:,None]
        value = np.sum(der, axis = 0)
        #print(beta)
        return value
# example
#x1=np.ones([4,3])
#x2 = np.arange(4)
#x1*x2[:,None]
# x2[:,np.newaxis]
    

    def GNR_stage2_partial_integral(self, gammas, df, poly):
        partial_int = gammas[0]*df[poly[0]] * df[poly[3]]
        partial_int = partial_int + gammas[1] * df[poly[1]] * df[poly[3]]
        partial_int = partial_int + gammas[2] * df[poly[2]] * df[poly[3]]
        partial_int = partial_int + (gammas[3]/2) * df[poly[3]]**2
        return partial_int

    def GNR_stage2_g_markov(self, delta, alphas, df , lag_poly):
        L_omega = df['L_phi'] + df[lag_poly].values@alphas
        s = delta[0] * L_omega
        # s = delta[0] * L_omega + delta[1] * L_omega**2
        return s

    def GNR_stage2_obj(self, pars, df, poly, lag_poly):
        ''' the LHS is phi'''
        alphas = pars[:-2]
        delta = pars[-2:]
        LHS = df['phi']
        RHS = - df[poly].values@alphas + self.GNR_stage2_g_markov(delta, alphas, df, lag_poly)
        
        return sum((LHS-RHS)**2)


    def GNR(self, df_input, ln_gross_output, statevar, flexiblevar, price_m, price_y, industry = None, stage2_polymax = 2, print_output = True):
        '''GNR production estimation'''

        stage1_polymax = 1
        # will put this back to an input argument. But now the integral function is not general enough
        # check GNR_stage2_partial_integral
        # also the optimization does not work with poly >= 2 !!!

        ids = [self.industry_id, self.firm_id, self.year_id]
        prices = [price_m, price_y]
        variables = [ln_gross_output] + statevar + [flexiblevar] + prices
        
        # finish checking input arguments
        if print_output:
            print('--GNR:')
            print('--for industry {} ----------------------'.format(industry))


        # STEP 0 : get data, generate non-par polynomials
        # 1.get the sample data
        if industry != None:
            df = df_input.loc[self.full_sample[self.industry_id]==industry, ids + variables]
        else:
            df = df_input[ids + variables]

        # 2. generate stage 1 dataframe and polynomials
        [df_stage1, poly] = mf.gen_poly(df,
                                        statevar+[flexiblevar],
                                        stage1_polymax, print_output = print_output)
        
        # --------------------------- 
        # STAGE 1
        # identify coeff of flexible var
        # ---------------------------

        # 1. generate share of flexiblevar to the total ouutput in $
        df_stage1['ls'] = df_stage1[flexiblevar] + df_stage1[price_m] - df_stage1[ln_gross_output] - df_stage1[price_y]

        # 2. estimate: use state variables to predit the share
        # (1) estimate
        beta_initial = np.ones(len(poly))*2
        constraint = lambda x: self.GNR_stage1_constraint(df_stage1,poly,x)
        results = opt.minimize(self.GNR_stage1_obj,
                                beta_initial,
                                args = (df_stage1,'ls',poly), 
                                constraints={'type': 'ineq', 'fun': constraint} )
        if print_output:
            print('--GNR: stage 1 optimization completed')


        # (2) get residul and fitted value
        gammas = results.x
        fitted = np.log(df_stage1[poly].values@gammas)
        eps = df_stage1['ls'] - fitted
        E = np.mean(np.exp(eps)) 

        # (3) plot the fitting
        if print_output:
            figpath = self.resultpath + '/' + 'GNR' 
            figname = 'GNR_stage1_fitting'
            self.plot_nonpar_fit(df_stage1['ls'], fitted, figpath,figname)
            print(poly) 

        # (4) adjust the gammas
        gammas = gammas / E

        # 3. get the material elasticity
        df_stage1['alpha_m']= df_stage1[poly]@gammas

        # 4. partial out the part correlated with flexible input
        # prepare for stage 2
        df_stage1['partialint_m'] = self.GNR_stage2_partial_integral(gammas, df_stage1, poly)
        df_stage1['phi'] = df_stage1[ln_gross_output] - df_stage1['partialint_m'] - eps
        if print_output:
            print('--GNR: stage 1 completed --------------------- ')

        # --------------------------- 
        # STAGE 2
        # identify coeff of state variables
        # for omega_it:
        #    1. use Markov: expectation = function of omega_i,t-1  
        #    2. use control function to get omega_i,t-1  in terms of statevar_i,t-1
        # ---------------------------
        
        # 0. get a new dataframe for stage 2 (because when taking lags, we lose obs.)
        df_stage2 = df_stage1.drop(columns=poly) # the polynomials were just for stage 1 nonpar estimation

        # 1. generate polynomials
        [df_stage2, poly] = mf.gen_poly(df_stage2,['ll','lc'], stage2_polymax, print_output = print_output)

        # 2. generate lags
        lag_vars = poly + ['phi']
        lag_vars_newname = ['L_' + x  for x in lag_vars] 
        lag_poly =  ['L_' + x  for x in poly] 
        df_stage2[lag_vars_newname] = df_stage2.groupby(self.firm_id)[lag_vars].shift(periods = 1)
        # drop a row if any element == nan
        df_stage2.dropna(inplace = True)

        # 3. optimize
        # (1) innitial value: 1
        alphas_initial = np.ones(len(poly))
        delta_initial = np.ones(2)
        pars = np.append(alphas_initial,delta_initial)
        # (2) optimize
        results = opt.minimize(self.GNR_stage2_obj,
                                pars,
                                args = (df_stage2, poly, lag_poly) )
        alphas = results.x[:-2]
        delta = results.x[-2:]
        # note, alpha is in the same order as poly
        if print_output:
            print('--GNR: stage 2 optimization completed')


        # (2) plot the fitting
        if print_output:
            figpath = self.resultpath + '/' + 'GNR' 
            figname = 'GNR_stage2_fitting'
            # note here it's a negative sign! By theory!
            df_stage2['omega_expected'] = self.GNR_stage2_g_markov(delta, alphas, df_stage2, lag_poly)
            df_stage2['partial_k_l'] = - df_stage2[poly].values@alphas
            fitted =  df_stage2['partial_k_l'] + df_stage2['omega_expected']
            self.plot_nonpar_fit(df_stage2['phi'], fitted, figpath,figname)
            print(poly)

        # 4. generate elasticities for k and l
        if stage2_polymax == 3:
            df_stage2['alpha_k'] = - (alphas[poly.index('poly_ll_lc_0_1')] + 
                                    2 * alphas[poly.index('poly_ll_lc_0_2')] * df_stage2['lc'] +
                                    3 * alphas[poly.index('poly_ll_lc_0_3')] * df_stage2['lc']**2 +
                                    alphas[poly.index('poly_ll_lc_1_1')] * df_stage2['ll'] +
                                    2 * alphas[poly.index('poly_ll_lc_1_2')] * df_stage2['ll'] * df_stage2['lc'] +
                                    alphas[poly.index('poly_ll_lc_2_1')] * df_stage2['ll']**2 )

            df_stage2['alpha_l'] = - (alphas[poly.index('poly_ll_lc_1_0')] + 
                                    2 * alphas[poly.index('poly_ll_lc_2_0')] * df_stage2['ll'] + 
                                    3 * alphas[poly.index('poly_ll_lc_3_0')] * df_stage2['ll']**2 +
                                    alphas[poly.index('poly_ll_lc_1_1')] * df_stage2['lc'] +
                                    2 * alphas[poly.index('poly_ll_lc_2_1')] * df_stage2['ll'] * df_stage2['lc'] +
                                    alphas[poly.index('poly_ll_lc_1_2')] * df_stage2['lc']**2 )

        if stage2_polymax == 2:
            df_stage2['alpha_k'] = - (alphas[poly.index('poly_ll_lc_0_1')] + 
                                    2 * alphas[poly.index('poly_ll_lc_0_2')] * df_stage2['lc'] +
                                    alphas[poly.index('poly_ll_lc_1_1')] * df_stage2['ll'] )
            df_stage2['alpha_l'] = - (alphas[poly.index('poly_ll_lc_1_0')] + 
                                    2 * alphas[poly.index('poly_ll_lc_2_0')] * df_stage2['ll'] + 
                                    alphas[poly.index('poly_ll_lc_1_1')] * df_stage2['lc'] )
        if stage2_polymax == 1:
            df_stage2['alpha_k'] = -alphas[poly.index('poly_ll_lc_0_1')]
            df_stage2['alpha_l'] = -alphas[poly.index('poly_ll_lc_1_0')]

        if print_output:
            print('--GNR: stage 2 completed ---------------------')
         
        # --------------------------- 
        # STAGE 3
        # estimate a fitted value for ln( gross output)
        # ---------------------------
        # 4. plot the fitted gross output
        if print_output:
            print('')
            print('--GNR: fitted ln(gross output) (single out productivities)')
            figpath = self.resultpath + '/' + 'GNR' 
            figname = 'GNR_stage2_fitting'
            # note here it's a negative sign! By theory!
            fitted_ly = df_stage2['partialint_m'] + df_stage2['partial_k_l']
            self.plot_nonpar_fit(df_stage2[ln_gross_output], fitted_ly, figpath,figname)

        # --------------------------- 
        # STAGE 4
        # export elasticities and the final dataframe
        # ---------------------------
        #df_stage2 = df_stage2.drop(columns=poly)

        alpha_m = df_stage2['alpha_m'].mean()
        alpha_k = df_stage2['alpha_k'].mean()
        alpha_l = df_stage2['alpha_l'].mean()

        alphas_output = {'alpha_k':alpha_k, 
                         'alpha_l':alpha_l, 
                         'alpha_m':alpha_m}

        # output stage1 and stage2 results for further checking
        parameters = [gammas, alphas, poly]

        if print_output:
            print('# --- ~ o(*￣▽￣*)o ~')
            print('# --- Complete GNR !!!  ~ o(*￣▽￣*)o ~')
        
        return [alphas_output,df_stage1,df_stage2, parameters]


    # IV. visualize  ------------------------------------------------------
    def plot_nonpar_fit(self, yvar, fitted,figpath,figname):

        # Figure parameters
        area = np.pi*3
        filename = figpath + '/' + figname + '.png'

        # Plot
        plt.figure()
        plt.scatter(fitted, yvar, s=area, alpha=0.5)
        plt.title('Scatter plot: non-parametric fitting')
        plt.xlabel('fitted value')
        plt.ylabel('y')
        plt.savefig(filename)
        plt.show()

        # 2 plot the histogram
        #histname = 'scatter_non'
        #plt.figure()
        #plt.hist(likelihood_c, density = True )
        #plt.xlabel('Logit Choice probability of effective choices')
        #plt.xlim(0,1)

        









