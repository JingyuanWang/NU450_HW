********************************************************************************
* HW2
* Purpose: Q2
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


replace fine = fine * 1000

********************************************************************************
* II. reg Q2 
********************************************************************************


capture eststo clear
* (a)
eststo: reg inspection lag_violator_nothpv lag_hpv_status dav ,  cluster(region_ind_code) 
eststo: reg inspection lag_violator_nothpv lag_hpv_status i.lag_violator_nothpv#c.dav i.lag_hpv_status#c.dav,  cluster(region_ind_code) 

* (b)
eststo: reghdfe inspection lag_violator_nothpv lag_hpv_status dav , a( i.region i.orig_naics i.gravity) cluster(region_ind_code) 
estadd local gravityFE "Yes"
estadd local regionFE "Yes"
estadd local naicsFE "Yes"
eststo: reghdfe inspection lag_violator_nothpv lag_hpv_status i.lag_violator_nothpv#c.dav i.lag_hpv_status#c.dav, a( i.region i.orig_naics i.gravity) cluster(region_ind_code) 
estadd local gravityFE "Yes"
estadd local regionFE "Yes"
estadd local naicsFE "Yes"


* (c)
eststo: reghdfe fine inspection violation lag_violator_nothpv lag_hpv_status dav , a( i.region i.orig_naics i.gravity) cluster(region_ind_code) 
estadd local gravityFE "Yes"
estadd local regionFE "Yes"
estadd local naicsFE "Yes"
eststo: reghdfe fine inspection violation lag_violator_nothpv lag_hpv_status i.lag_violator_nothpv#c.dav i.lag_hpv_status#c.dav, a( i.region i.orig_naics i.gravity) cluster(region_ind_code) 
estadd local gravityFE "Yes"
estadd local regionFE "Yes"
estadd local naicsFE "Yes"

estout,  ///
        cells(b(star fmt(3)) se(par fmt(3)))  ///
        modelwidth(10) varwidth(10) /// 
		stats(r2_a N gravityFE naicsFE regionFE, fmt(%9.3f %9.0g) labels(R-squared)) 
		

estout using "$results/Q2.tex" , style(tex) ///
        cells(b(star fmt(3)) se(par fmt(3)))  ///
        modelwidth(10) varwidth(10) /// 
		stats(r2_a N gravityFE naicsFE regionFE, fmt(%9.3f %9.0g) labels(R-squared)) ///
		drop( _cons ) ///
		replace
















