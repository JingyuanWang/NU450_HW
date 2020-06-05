********************************************************************************
* HW2
* Purpose: Q4
* 
* Author: Jingyuan Wang
********************************************************************************



********************************************************************************
* I. import data
********************************************************************************

import delimited using "$data/analysis_data.csv", clear

********************************************************************************
* II. basic clean
********************************************************************************

* ------------ 1. basic clean ---------------------
* 1. time id and xtset 
rename quarter quarter_str
gen year = substr(quarter_str, 1, 4)
gen quarter = substr(quarter_str, -1, 1)
destring year, replace
destring quarter, replace
order year quarter, b(quarter_str)
gen time_id = 4*(year-2007) + quarter

xtset frsnumber time_id

gen lag_ordered_violator = L.ordered_violator
* ------------ 2. pick one good region*naics*gravity -----------------


foreach j in 22 32 202 249 {
preserve
sort frsnumber time_id
keep if omega1 == `j'
drop if lag_investment == . | lag2_investment == . | lag_ordered_violator == .

*gen lag_investment_counterfactual = 1 - lag_investment
*replace lag_investment_counterfactual = 0 if lag_compliance == 1 

export delimited using "$data/analysis_data_omega1_`j'.csv", replace 
restore 
}
