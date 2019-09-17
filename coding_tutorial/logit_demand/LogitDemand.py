import numpy as np
import pandas as pd
import scipy.stats as stats

# ------------------------------------------------------------------------
# NOTE
# ------------------------------------------------------------------------
# Purpose:
# define class: LogitDemand, with method:
#     1. choice_probability(I,X,beta)
#     2. loglikelihood(Y,I,X,beta)
#
# Definition of several variables in this file:
# n: number of consumers
# m: number of products
#    outside option normalized to 0 and not included among the choices
# k: length of beta, number of variables
# ------------------------------------------------------------------------

class LogitDemand:
    datatype = 'a dataframe storing ids, Y(choices), X(attributes)'

    # get the main dataframe (and get the set of variables)
    def __init__(self, df):
        self.main = df
        self.columns = df.columns.tolist()

    # I. set variables ----------------------------------------------------
    # 1. ids
    def set_ids(self, case_id, choice_id, case_groupid = None, choice_groupid = None):
        
        # (1) covert ids into str
        if isinstance(case_id, list):
            if len(case_id) == 1:
                case_id = case_id[0]
            else:
                raise Exception('Please input only 1 variable')
        if isinstance(choice_id, list):
            if len(choice_id) == 1:
                choice_id = choice_id[0]
            else:
                raise Exception('Please input only 1 variable')
        # To list:
        if isinstance(case_groupid, str):
            case_groupid = [case_groupid]
        if isinstance(choice_groupid, str):
            choice_groupid = [choice_groupid]

        # (2) set ids
        self.case_id = case_id
        self.case_groupid = case_groupid
        self.choice_id = choice_id
        self.choice_groupid = choice_groupid

        # (3) check the uniqueness:
        # a. required to be unique
        # check case_id * choice_id uniquely define the data
        dup = self.main.groupby([case_id,choice_id]).size().rename('dup')
        problematic_ids = dup[dup>1].index.unique().to_numpy()
        if len(problematic_ids) != 0:
            self.problematic_ids = problematic_ids
            raise Exception('''case_id ({}) * choice_id ({}) not unique in the dataframe.
                Return: self.problematic_ids, a series of id tuples which have more than 1 obs in the dataframe'''.format(case_id[0],choice_id[0]))
        
        # sort according to case-choice
        self.main = self.main.sort_values([case_id, choice_id]).reset_index(drop = True)

        # b. can be not uniquely identified, but need to raise it


    # 2. variable related to product and to consumer
    def set_attributes(self, prod_varnames, consumer_varnames, consumer_prod_varnames):
        # prod_varnames should include price

        # To list, even if contains only 1 variable:
        if isinstance(prod_varnames, str):
            prod_varnames = [prod_varnames]
        if isinstance(consumer_varnames, str):
            consumer_varnames = [consumer_varnames]
        if isinstance(consumer_prod_varnames, str):
            consumer_prod_varnames = [consumer_prod_varnames]

        # set:
        self.prod_varnames = prod_varnames
        self.consumer_varnames = consumer_varnames
        self.consumer_prod_varnames = consumer_prod_varnames

    # 3. Y
    def set_Y(self,Y_name):
        # (1) check input type prefered input type: str
        if isinstance(Y_name,list):
            if len(Y_name) > 1:
                raise Exception('Please input only 1 variable')
            else:
                Y_name = Y_name[0]
        if not isinstance(Y_name, str):
            raise Exception('variable name should be str')
        if not (Y_name in self.columns):
            raise Exception('{} not in the DataFrame'.format(Y_name))
        else:
            Y = self.main[Y_name]

        # (2) check Y variable: 
        # if Y is not binary
        #     generate a binary variable Y_new = (Y==choice_id)
        if not Y.apply(lambda x: x in [0,1]).all():
            # if Y are bool, int, float (0,1)s, this will return not(Ture) = False.
            # The only True case is: Y not binary, no matter what datatype it is.

            # when Y is not binary, generate 
            new_Y_name = Y_name + '_new'
            self.main[new_Y_name] = (self.main[Y_name] == self.main[self.choice_id]) * 1
            Y_name = new_Y_name
            Y = self.main[Y_name]

        # Now Y is binary
        #     check: each case choce 1 and only 1 product
        #     implicitly check the old Y \in choice_id, if Y were not binary
        Y = Y * 1  # incase it's bool
        # check 1 and only 1 are selected
        total_selection = self.main.groupby(self.case_id).agg({Y_name: np.sum})
        tol = 10**(-10)
        if len(total_selection.loc[abs(total_selection[Y_name] -1) > tol]) != 0:
            self.total_selection_eachcase = total_selection
            raise Exception('''total_selection_eachcase != 1. 
                        Return: self.total_selection_eachcase, each row = each case.''')
        
        # (3) set Y:
        self.Y_name = Y_name



    # II. set analysis data: Y, regressor, I = case id --------------------
    def set_regressor(self, list_of_regressor):
        self.regressor = list_of_regressor

    # III. estimate MLE ---------------------------------------------------
    def choice_probability(self,beta):
        # Purpose: compute the probability of each consumer making each choices
        # Inputs:
        # I: vector of consumer index, n*c-by-1
        # X: matrix of consumer choice attributes, n*c-by-m, index = consumer_id
        # beta: coefficients, m-by-1
        I = self.main[self.case_id]
        X = self.main[self.regressor]

        # 1. check input data ----------------------------------------------------------------------
        # suppose the column size of X is correct
        nc,k = X.shape 
        # check X
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if not isinstance(X, pd.DataFrame):
            raise Exception("X should be a dataframe or an array")

        # check beta
        # (1) data type
        if isinstance(beta,list) or isinstance(beta,pd.Series):
            beta = np.array(beta)
        if not isinstance(beta, np.ndarray):
            raise Exception("beta should be a list, a series, or an array")
        # (2) check the size of beta = (k,1)
        if beta.shape == (k,):
            beta = beta.reshape((k,1))
        if beta.shape != (k,1):
            raise Exception("size of beta should be ({},1) or ({},) instead of {}".format(k, k, beta.shape))

        # check index I
        if I.shape != (nc,1) and I.shape != (nc,):
            raise Exception("size of I should be ({},1) or ({},) instead of {}".format(nc, nc, I.shape))

        # 2. calculate the probability --------------------------------------------------------------
        # (0) set up a dataframe to work with
        df = X.copy()
        df.index.name = 'index'
        df = df.reset_index()
        # use variable 'index' to track the order
        df['case_id'] = I
        
        # (1). give a score to each choice for each consumer
        df['score'] = X.values.dot(beta)
        # later on we will do \[prob = exp(score)/sum( exp(score)  of all product offered) \] for each person
        # to avoid exp(something large) = inf:
        #     for each person, 
        #         devide the denominator and numerator by exp(max product score for this person)
        #         equivalent to 
        #         score for each product - max score 
        maxscore = df.groupby('case_id').agg({'score':np.max}).rename(columns = {'score':'max_score'})
        df = pd.merge(df,maxscore, how = 'left', left_on = 'case_id', right_on = 'case_id')
        df['score'] = df['score'] - df['max_score']
        df.drop(columns = 'max_score')
        
        # (2). calculate a probability of chosing each product for each consumer, based on the score
        df['expscore'] = np.exp(df['score'])
        total_expscore = df.groupby('case_id').agg({'expscore': np.sum }).rename(columns = {'expscore':'total_expscore'})
        df = pd.merge(df, total_expscore, how = 'left', left_on = 'case_id', right_on = 'case_id')
        df['prob'] = np.exp(df['score'])/df['total_expscore']
        
        # 3. check and return values ---------------------------------------------------------------
        # check whether prob for each person sum up to 1
        total_prob = df.groupby('case_id').agg({'prob': np.sum})
        tol = 10**(-10)
        if len(total_prob.loc[abs(total_prob['prob'] -1) > tol]) != 0:
            self.problem_df = df
            self.problem_total_prob = total_prob
            print("probability does not sum up to 1 (or because of NaN/inf), return the dataframe to self.problem_df, self.problem_total_prob")
        
        # make sure the output probability is in the order as the main dataframe
        df.sort_values('index', inplace = True)
        probability = df['prob']
        if len(probability) != len(I):
            raise Exception("size of probability should be the same as I {}-by-1, current value {}-by-1".format(I.shape,I.shape))   

        return probability
        # return a vector of probablities, same size as I (n*k-by-1)


    def loglikelihood(self,beta):
        # Purpose: compute the NEGATIVE likelihood of a binary vector of choices Y, conditional on X
        # Inputs:
        # Y: binary-vector of choices, n*c-by-1
        # I: vector of consumer index, n*c-by-1
        # X: matrix of consumer choice attributes, n*c-by-m
        # beta: coefficients, m-by-1
        I = self.main[self.case_id]
        X = self.main[self.regressor]
        Y = self.main[self.Y_name]
       
        # 1. check input data ----------------------------------------------------------------------
        # suppose the column size of X is correct
        nc,k = X.shape 
        # check X
        if not np.logical_or(isinstance(X, pd.DataFrame), isinstance(X, np.ndarray)):
            raise Exception("X should be a dataframe or an array")
        
        # check beta
        # (1) data type
        if isinstance(beta,list) or isinstance(beta,pd.Series):
            beta = np.array(beta)
        if not isinstance(beta, np.ndarray):
            raise Exception("beta should be a list, a series, or an array")
        # (2) check the size of beta = (k,1)
        if beta.shape == (k,):
            beta = beta.reshape((k,1))
        if beta.shape != (k,1):
            raise Exception("size of beta should be ({},1) or ({},) instead of {}".format(k, k, beta.shape))
        
        # check Y  
        if Y.shape != (nc,1) and Y.shape != (nc,):
            raise Exception("size of Y should be ({},1) or ({},) instead of {}".format(nc, nc, Y.shape))
        
        # check index I
        if I.shape != (nc,1) and I.shape != (nc,):
            raise Exception("size of I should be ({},1) or ({},) instead of {}".format(nc, nc, I.shape))

        
        # 2. calculate loglikelihood -------------------------------------------------------------------
        precision = 10**-300
        
        # 2.1 likelihood for each obs = consumer-product
        likelihood_c_j = self.choice_probability(beta)
        
        # 2.2 likelihood for each consumer to make the observed choice
        likelihood_c_j.reset_index(drop = True, inplace = True)
        likelihood_c = likelihood_c_j.loc[Y==1]
        
        # [Adjustment 1] Avoid -inf:
        # If there is a choice probability almost = 0.0
        #         then the loglikelihood for this obs = -inf
        # then the summation of all obs == -inf, no matter how the other obs performs. This obs is weighted 100%.
        # to avoid this, remove all the 0s and replace to the precision or the minimum nonzero value.
        likelihood_c[likelihood_c<=precision] = precision
        
        # 2.3 summation of loglikelihood for all obs
        loglikelihood = -np.sum(np.log(likelihood_c))
        

        # 3. return ------------------------------------------------------------------------------------
        return loglikelihood
        # return a number, the NEGATIVE likelihood y|x

    # IV. visualize  ------------------------------------------------------
    def plot_fit(self,beta):

        Y = self.main[self.Y_name]
        # 1 estimate likelihood for each obs = consumer-product
        likelihood_c_j = self.choice_probability(beta)
        likelihood_c_j.reset_index(drop = True, inplace = True)
        likelihood_c = likelihood_c_j.loc[Y==1]




