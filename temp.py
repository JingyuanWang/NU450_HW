

# generate polynomials
def gen_poly(df_input, varnames, poly_max, print_output = True):
    '''given a dataframe and a list of variables:
    generate polynomials of the variables (up to (poly_max)th order) and save to the dataframe
    
    return: the new dataframe and a list of polynomial variable names'''
    
    df = df_input.copy()
    n = len(varnames)
    
    # specify group name: all the variable names start with groupname. 
    # example: poly_var1_var2_1_3 = var1^{1} \times var2^{3}
    groupname = 'poly'
    for i in range(n):
        groupname = groupname + '_' + varnames[i]
    
    
    # STEP 0 constant:
    polynames = '_0'*n
    new_polyvar_name = groupname + polynames
    df[new_polyvar_name] = 1
    if print_output:
        print('-- generate: {} --'.format(new_polyvar_name))
    
    # STEP 1 just 1 variable, 1st order-highest order: variable it self
    for i in range(n):
        for j in range(1,poly_max+1): 
            polynames = '_0'*(i) + '_' + str(j) + '_0' *(n-i-1)
            new_polyvar_name = groupname + polynames
            df[new_polyvar_name] = df[varnames[i]].values**poly_max
            if print_output:
                print('-- generate: {} --'.format(new_polyvar_name))
    
    # STEP 2 poly 2-poly_max, with number of variables >=2
    lenth = df.shape[0]
    
    for R in range(2,poly_max+1):
    	# from now on: fix the total polynomial order: R = poly_max
    	# pick n_include from all n variables

        for n_include in range(2,n+1):
        	# from now on fix number of variables this in iter.
        	# the rest steps:
        	# -- pick the n_include variables
        	# -- assign them poly orders add up to R


            # (1) pick n_include variables from all n variables
            for comb_of_vars in (iter.combinations(range(0,n), n_include)):
            	# from now on: fix n_include certain variables

                varnames_include = [varnames[comb_of_vars[0]]]
                for i in range(1,n_include):
                    varnames_include = varnames_include + [varnames[comb_of_vars[i]]]
                if print_output:
                    print('-- {}'.format(varnames_include))
            
                # (2) give each included variable a power
                cumul_list = list(iter.combinations(range(1,R), n_include-1))
                # For number of variables == n_include, total number of unique combinations = len(cumul_list)
                # each combination represents one new polynomial variable, generate them 1-by-1

                for i in range(len(cumul_list)):
                
                    # start to generate 1 new poly var
                    cumul = list(cumul_list[i])
                    power = list(cumul)
                    power[1:] = [cumul[i + 1] - cumul[i] for i in range(len(cumul)-1)]
                    power.append(R-cumul[-1])
                
                    output = np.zeros(lenth)
                    new_var_name = ''
                    polynames = ''
                    for var in range(n_include):
                        output = output + df[varnames_include[var]].values**power[var]
                        new_var_name = new_var_name + varnames_include[var] + '_'
                        polynames = polynames + '_' + str(power[var])
                    
                    new_polyvar_name = 'poly_' + new_var_name[:-1] + polynames
                    df[new_polyvar_name] = output
                    if print_output:
                        print('-- generate: {} --'.format(new_polyvar_name))
                    # end of generating 1 new poly var
                
                # end of generating all variable with fixed variable combination
            # end of loop over different combinations with fixed n_include
        # end of loop over n_include from 2-n, with fixed total polynomial orders

    if print_output:
    	print('# --- Finish generating polynomials ---')
    polynames = [n for n in df.columns.values if n.startswith('poly')]
    
    return [df, polynames]
