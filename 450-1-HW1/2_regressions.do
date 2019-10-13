

********************************************************************************
* 450-1-HW1
* Purpose:
* 
* Author: Jingyuan Wang
********************************************************************************


* Parameters *******************************************************************

local balance = 0
local industry = 13

********************************************************************************
* 1. import the cleaned data file
********************************************************************************
use "$data/PS1_data_cleaned.dta", clear
xtset firm_id year
keep if industry == `industry'

* change to a balanced panel
local panel "fullpanel"
if `balance' == 1 {
	sort firm_id industry //reason for doing this: some firms like #347 are categorized into different industries in different years, we drop them
	by firm_id industry: gen obs_sameidn = _N
	keep if obs_sameidn==10
	drop obs*
	local panel "balancedpanel"
}


********************************************************************************
* 2. compare the following 5 regressions
********************************************************************************

* 0. prepare for log long diff
xtset firm_id year
sort firm_id year
by firm_id: gen ly5 = ly[_n-5]
by firm_id: gen lc5 = lc[_n-5]
by firm_id: gen ll5 = ll[_n-5]
by firm_id: gen lm5 = lm[_n-5]


* Gross Output
{
local y = "ly"
    
	* get dependent variable name
	local dependentvar: variable label `y'
	
	est clear
	* (1)ols
	eststo:  reg ly lc ll lm, r
	* (2) first difference
	eststo: reg d.ly d.lc d.ll d.lm, nocons
	* (3) long difference
	eststo: reg ly5 lc5 ll5 lm5, nocons
	* (4) fixed effect
	eststo: xtreg ly lc ll lm, fe
	estimate store fixed
	* (5) random effect
	eststo: xtreg ly lc ll lm, re
	estimate store random
    
	disp("`dependentvar'")
	estout using "$results/Reg_`panel'_`y'.tex" , ///
	    cells(b(star fmt(3)) se(par fmt(3)))  ///
		modelwidth(13) varwidth(13) /// 
		stats(r2_a N, fmt(%9.3f %9.0g) labels(R-squared)) ///
		mlabels("OLS" "First Diff" "Long Diff" "FE" "RE" )	///
		style(tex) varlabels(_cons const) rename(D.ll ll D.lc lc D.lm lm lc5 lc ll5 ll lm5 lm) replace
	* hausman test 
	hausman fixed random

}

* Value Added
{
local y = "ly_va"
    
	* get dependent variable name
	local dependentvar: variable label `y'
	
	est clear
	* (1)ols
	eststo:  reg ly lc ll , r
	* (2) first difference
	eststo: reg d.ly d.lc d.ll, nocons
	* (3) long difference
	eststo: reg ly5 lc5 ll5, nocons
	* (4) fixed effect
	eststo: xtreg ly lc ll, fe
	estimate store fixed
	* (5) random effect
	eststo: xtreg ly lc ll, re
	estimate store random
	
    disp("`dependentvar'")
	estout using "$results/Reg_`panel'_`y'.tex", ///
	    cells(b(star fmt(3)) se(par fmt(3)))  ///
		modelwidth(13) varwidth(13) /// 
		stats(r2_a N, fmt(%9.3f %9.0g) labels(R-squared)) ///
		mlabels("OLS" "First Diff" "Long Diff" "FE" "RE" )	///
		style(tex) varlabels(_cons const) rename(D.ll ll D.lc lc lc5 lc ll5 ll) replace
	* hausman test
	hausman fixed random

}















