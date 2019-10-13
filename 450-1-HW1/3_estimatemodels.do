
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

est sto
* Arellano & Bond
by firm_id: gen ly1 = ly[_n-1]
by firm_id: gen ly1_va = ly_va[_n-1]
by firm_id: gen lc1 = lc[_n-1]
by firm_id: gen ll1 = ll[_n-1]
by firm_id: gen lm1 = lm[_n-1]
xtabond2 ly_va lc ll,    gmmstyle(lc ll)    ivstyle(ly1_va lc1 ll1,     equation(level)) h(1) twostep robust small
xtabond2 ly    lc ll lm, gmmstyle(lc ll lm) ivstyle(ly1    lc1 ll1 lm1, equation(level)) h(1) twostep robust small

* Blundell & Bond
xtdpdsys ly_va L(0/2).(lc ll),    vce(robust)
est sto: xtdpdsys ly    L(0/2).(lc ll lm), vce(robust)


* 1'. OP command *************************************************************

est sto:  opreg ly_va, exit(exit) state(lc) proxy(linv) free(ll) vce(bootstrap, seed(1) rep(2))

* 2. LP  *********************************************************************
* LP
est sto: levpet ly_va, free(ll) proxy(lm) capital(lc) valueadded

* ??? what's the difference between this and opreg?
est sto:  opreg ly_va, exit(exit) state(lc) proxy(lm) free(ll) vce(bootstrap, seed(1) rep(2))

* 3. ACF *********************************************************************
est sto: acfest ly_va, free(ll) state(lc) invest proxy(linv) va

estout , ///
	    cells(b(star fmt(3)) se(par fmt(3)))  ///
		modelwidth(13) varwidth(13) /// 
		stats(r2_a N, fmt(%9.3f %9.0g) labels(R-squared)) ///
		varlabels(_cons const)  replace

















