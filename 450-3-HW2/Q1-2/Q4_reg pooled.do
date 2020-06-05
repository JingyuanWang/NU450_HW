

********************************************************************************
* HW2
* Purpose: Q4
* 
* Author: Jingyuan Wang
********************************************************************************



********************************************************************************
* 2. import data
********************************************************************************
import delimited using "$data/analysis_data_omega1_22_readyforreg.csv", clear
tempfile this_omega 
foreach j in 32 202 249 {
	preserve
		import delimited using "$data/analysis_data_omega1_`j'_readyforreg.csv", clear
		save `this_omega.dta', replace 
	restore
	append using `this_omega'
}


********************************************************************************
* 1. set parameter
********************************************************************************


local actual_inv_t = 1
local counterfactual_inv_t = 0




capture eststo clear
foreach actual_inv_t in 0 1 {
	foreach counterfactual_inv_t in 0 1 {

********************************************************************************
* 2. generate var
********************************************************************************
capture drop inv Y 
capture drop diff*
gen inv = 1
replace inv = -1  if lag_investment_actual == 1
if `actual_inv_t' == 1{
    replace inv = inv - 0.95 if ordered_violator_actual != 0 
}
if `counterfactual_inv_t' == 1{
    replace inv = inv + 0.95 if state_var_id_counterfactual != 0 | state_var_id_counterfactual != 3
}


gen Y = lnp0 - lnp1 
replace Y = lnp1 - lnp0  if lag_investment_actual == 1
replace Y = Y + 0.95 * lnp`actual_inv_t'_actual - 0.95 * lnp`counterfactual_inv_t'_counterfactual 

* transition to t+1:

foreach var in compliance0 violator_nothpv0 hpv_status0 compliance1 violator_nothpv1 hpv_status1 {
	capture drop `var'_a `var'_cf
	gen `var'_a = `var'_`actual_inv_t'_a
	gen `var'_cf = `var'_`counterfactual_inv_t'_cf 
	gen diff_T_`var' = `var'_a - `var'_cf
}

foreach var in compliance violator_nothpv hpv_status  {
    gen diff2_T_`var' = diff_T_`var'0 + diff_T_`var'1
}


********************************************************************************
* 3. reg
********************************************************************************


eststo: reg Y inv diff2_T*
eststo: reg Y inv diff_T*
estadd local heterogeneity "Yes"

	}
}

********************************************************************************
* III. save 
********************************************************************************
estout,  ///
        cells(b(star fmt(3)) se(par fmt(3)))  ///
        modelwidth(10) varwidth(10) /// 
		mlabels("00" "00" "01" "01" "10" "10" "11" "11") ///
		stats(r2_a N heterogeneity, fmt(%9.3f %9.0g) labels(R-squared)) 

estout using "$results/Q4_pooled_a`actual_inv_t'_cf`counterfactual_inv_t'.tex" , style(tex) ///
        cells(b(star fmt(3)) se(par fmt(3)))  ///
        modelwidth(10) varwidth(20) /// 
		mlabels("00" "00" "01" "01" "10" "10" "11" "11") ///
		stats(r2_a N heterogeneity, fmt(%9.3f %9.0g) labels(R-squared)) ///
		replace
		
		
		
		
		
		
		
