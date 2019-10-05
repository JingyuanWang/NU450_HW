
clear all
********************************************************************************
* 450-1-HW1
* Purpose:
* 
* Author: Jingyuan Wang
********************************************************************************


* Parameters *******************************************************************

local industry = 1

********************************************************************************
* 1. import the cleaned data file
********************************************************************************
use "$data/PS1_data_cleaned.dta", clear

xtset firm_id year
keep if industry == `industry'

********************************************************************************
* 2. 
********************************************************************************

* 1. OP Munual ****************************************************************
{
* (0) local name for 4th order polynomials
local poly_lc_linv = "linv lc c.linv#c.lc c.linv#c.linv c.lc#c.lc "
* 3rd order
local poly_lc_linv = "`poly_lc_linv' c.lc#c.lc#c.lc c.linv#c.linv#c.linv c.linv#c.lc#c.lc c.linv#c.linv#c.lc " 
* 4th order
*local poly_lc_linv = "`poly_lc_linv' c.linv#c.linv#c.linv#c.linv c.linv#c.linv#c.linv#c.lc c.linv#c.linv#c.lc#c.lc c.linv#c.lc#c.lc#c.lc c.lc#c.lc#c.lc#c.lc " 


* Step I - regress lny on variable inputs and 
regress ly_va ll `poly_lc_linv' ,  vce(cluster firm_id)
predict ly_va_hat if e(sample), xb
scalar b_ll = _b[ll]
scalar cons = _b[_cons]

* Step II -- Estimate probability of survival 
probit exit L.(`poly_lc_linv') 
predict p_hat if e(sample), pr

* Step III -- Nonlinear regression of y - lnl*b_lnl 
* (1) get phi_hat (whether to exclude a constant does not matter: anyways, beta0 is not identified)
gen phi_hat = ly_va_hat - ll*b_ll 

* (2) Next, generate the depvar for the nonlinear equation
* phi = Output minus the contributions of the variable inputs
gen LHS = ly_va - ll*b_ll 

* (3) Reg
* mark out missing observations
generate useme = 1
gen l1phi = L.phi_hat
gen l1lc = L.lc

foreach var of varlist LHS lc l1phi l1lc  {
	replace useme = 0 if `var' >= .
}

* nonlinear reg
nl ( LHS = {b0} + {bk}*lc +  			///
	        {t1}*(l1phi - {bk}*l1lc - {b0}) +		///
	        {t1_sq}*(l1phi - {bk}*l1lc - {b0})^2 +	 ///
			{t1_poly3}*(l1phi - {bk}*l1lc - {b0})^3 +		///
	        {t2}*p_hat + ///
			{t2_sq}*p_hat^2 + ///
			{t2_poly3}*p_hat^3 + ///
	        {t1t2}*(l1phi - {bk}*l1lc - {b0})*p_hat  ) 	///
	if useme , vce(cluster firm_id)
nl ( phi_hat = {b0} + {bk}*lc +  			///
	        {t1}*(l1phi - {bk}*l1lc - {b0} )+		///
	        {t1_sq}*(l1phi - {bk}*l1lc - {b0})^2 +	 ///
			{t1_poly3}*(l1phi - {bk}*l1lc - {b0})^3 +		///
	        {t2}*p_hat + ///
			{t2_sq}*p_hat^2 + ///
			{t2_poly3}*p_hat^3 + ///
	        {t1t2}*(l1phi - {bk}*l1lc - {b0})*p_hat  ) 	///
	if useme , vce(cluster firm_id)
disp("bl")
disp(b_ll)
}
* 1'. OP ***********************************************************************

opreg ly_va, exit(exit) state(lc) proxy(linv) free(ll) vce(bootstrap, seed(1) rep(2))

*opreg ly, exit(exit) state(lc) proxy(linv) free(ll lm) vce(bootstrap, seed(1) rep(2))




