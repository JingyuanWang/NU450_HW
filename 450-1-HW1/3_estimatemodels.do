
clear all
********************************************************************************
* 450-1-HW1
* Purpose:
* 
* Author: Jingyuan Wang
********************************************************************************


* Parameters *******************************************************************

local industry = 13

********************************************************************************
* 1. import the cleaned data file
********************************************************************************
use "$data/PS1_data_cleaned.dta", clear

xtset firm_id year
keep if industry == `industry'

********************************************************************************
* 2. Estimate Models Arellano and Blundell and Bonf
********************************************************************************

est clear
* Arellano & Bond
by firm_id: gen ly1 = ly[_n-1]
by firm_id: gen ly1_va = ly_va[_n-1]
by firm_id: gen lc1 = lc[_n-1]
by firm_id: gen ll1 = ll[_n-1]
by firm_id: gen lm1 = lm[_n-1]
xtabond2 ly_va lc ll,    gmmstyle(lc ll)    ivstyle(ly1_va lc1 ll1,     equation(level)) h(1) twostep robust small
estimates store AB_va
xtabond2 ly    lc ll lm, gmmstyle(lc ll lm) ivstyle(ly1    lc1 ll1 lm1, equation(level)) h(1) twostep robust small
estimates store AB_go

* Blundell & Bond
xtdpdsys ly_va L(0/2).(lc ll),    vce(robust)
estimates store BB_va
xtdpdsys ly    L(0/2).(lc ll lm), vce(robust)
estimates store BB_go

* 1. OP Munual ***************************************************************
{
* Step 0 - Local name for 4th order polynomials
local poly_lc_linv = "linv lc c.linv#c.lc c.linv#c.linv c.lc#c.lc "
* 3rd order
local poly_lc_linv = "`poly_lc_linv' c.lc#c.lc#c.lc c.linv#c.linv#c.linv c.linv#c.lc#c.lc c.linv#c.linv#c.lc " 
* 4th order
*local poly_lc_linv = "`poly_lc_linv' c.linv#c.linv#c.linv#c.linv c.linv#c.linv#c.linv#c.lc c.linv#c.linv#c.lc#c.lc c.linv#c.lc#c.lc#c.lc c.lc#c.lc#c.lc#c.lc " 


* Step I - regress lny on variable inputs and 
regress ly_va ll `poly_lc_linv' ,  vce(cluster firm_id)
predict ly_va_hat if e(sample), xb
scalar b_ll = _b[ll]

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

* 1'. OP command *************************************************************
opreg ly_va, exit(exit) state(lc) proxy(linv) free(ll) vce(bootstrap, seed(1) rep(2))
estimates store OP

* 2. LP  *********************************************************************
* LP
levpet ly_va, free(ll) proxy(lm) capital(lc) valueadded
estimates store LP

* ??? what's the difference between this and opreg?
opreg ly_va, exit(exit) state(lc) proxy(lm) free(ll) vce(bootstrap, seed(1) rep(2))


* 3. ACF *********************************************************************
acfest ly_va, free(ll) state(lc) invest proxy(linv) va
estimates store ACF

esttab AB_va AB_go BB_va BB_go OP LP ACF

















