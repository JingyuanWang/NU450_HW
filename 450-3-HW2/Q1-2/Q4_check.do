********************************************************************************
* HW2
* Purpose: Q4: finding termination action
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

rename quarter quarter_str
gen year = substr(quarter_str, 1, 4)
gen quarter = substr(quarter_str, -1, 1)
destring year, replace
destring quarter, replace
order year quarter, b(quarter_str)

gen time_id = 4*(year-2007) + quarter

xtset frsnumber time_id

egen region_ind_code = group(region orig_naics)



* generate lags
gen lag_ordered_violator = L.ordered_violator
gen lag2_ordered_violator = L2.ordered_violator
forvalues i = 3(1)5{
	gen lag`i'_investment = L`i'.investment 
	gen lag`i'_ordered_violator = L`i'.ordered_violator 
}

********************************************************************************
* III. basic stats
********************************************************************************

* 1) are there plants never violating?
bysort frsnumber: egen total_violation = total(violator)
bysort frsnumber: egen total_hpv = total(hpv_status)

unique frsnumber if total_violation == 0
unique frsnumber if total_violation == 27
unique frsnumber if total_hpv == 27

sum investment if total_violation == 27
sum investment if total_hpv == 27

* 2) are there plants never invest?
bysort frsnumber: egen total_investment = total(investment)
unique frsnumber if total_investment == 0
unique frsnumber if total_investment == 27

********************************************************************************
* III. find out termination action set
********************************************************************************
local choice = 0
local condition = " & total_violation != 0 "
forvalues i = 1(1)20{
disp("#-- current state: omega1 == `i' --------------------------------------- ")
disp("keep 2 period")
tab ordered_violator lag2_ordered_violator  if omega1 == `i' `condition' & lag_investment == `choice' & lag2_investment == `choice' 
disp("keep 3 period")
tab ordered_violator lag3_ordered_violator  if omega1 == `i' `condition' & lag_investment == `choice' & lag2_investment == `choice' & lag3_investment == `choice'
disp("keep 4 period")
tab ordered_violator lag4_ordered_violator  if omega1 == `i' `condition' & lag_investment == `choice' & lag2_investment == `choice' & lag3_investment == `choice' & lag4_investment == `choice' 
disp("keep 5 period")
tab ordered_violator lag5_ordered_violator  if omega1 == `i' `condition' & lag_investment == `choice' & lag2_investment == `choice' & lag3_investment == `choice' & lag4_investment == `choice' & lag5_investment == `choice'
}

disp("#-- all plants --------------------------------------------------------- ")
disp("keep 2 period")
tab ordered_violator lag2_ordered_violator  if lag_investment == `choice' `condition' & lag2_investment == `choice' 
disp("keep 3 period")
tab ordered_violator lag3_ordered_violator  if lag_investment == `choice' `condition' & lag2_investment == `choice' & lag3_investment == `choice'
disp("keep 4 period")
tab ordered_violator lag4_ordered_violator  if lag_investment == `choice' `condition' & lag2_investment == `choice' & lag3_investment == `choice' & lag4_investment == `choice' 
disp("keep 5 period")
tab ordered_violator lag5_ordered_violator  if lag_investment == `choice' `condition' & lag2_investment == `choice' & lag3_investment == `choice' & lag4_investment == `choice' & lag5_investment == `choice'







