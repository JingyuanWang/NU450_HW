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


gen lag_dav = L.dav
gen lag_ordered_violator = L.ordered_violator
gen lag_violator = L.violator
********************************************************************************
* II. reg Q3 a 
********************************************************************************


capture eststo clear		
eststo: reghdfe fine  lag_investment  lag_hpv_status  lag_dav ///
                      if lag_compliance == 0 , a( i.region i.orig_naics i.gravity) cluster(region_ind_code) 
estadd local gravityFE "Yes"
estadd local regionFE "Yes"
estadd local naicsFE "Yes"		
			  
eststo: reghdfe fine  lag_investment  lag_hpv_status lag_dav lag2_investment ///
                      if lag_compliance == 0 , a( i.region i.orig_naics i.gravity) cluster(region_ind_code) 	
estadd local gravityFE "Yes"
estadd local regionFE "Yes"
estadd local naicsFE "Yes"		
		
eststo: reghdfe fine  lag_investment  lag_hpv_status  ///
                i.lag_hpv_status#c.lag_dav  ///
				i.lag_investment#i.lag_hpv_status#c.lag_dav  if lag_compliance == 0 , a(region orig_naics gravity ,save) cluster(region_ind_code) 
estadd local gravityFE "Yes"
estadd local regionFE "Yes"
estadd local naicsFE "Yes"
				
estout,  ///
        cells(b(star fmt(3)) se(par fmt(3)))  ///
        modelwidth(10) varwidth(20) /// 
		stats(r2_a N gravityFE naicsFE regionFE, fmt(%9.3f %9.0g) labels(R-squared)) 

********************************************************************************
* II. reg Q3 b predict fine values
********************************************************************************		

* 1. reg
reg fine lag_investment  lag_hpv_status  ///
				i.lag_investment#i.lag_hpv_status#c.lag_dav  i.region i.orig_naics i.gravity if lag_compliance == 0

* 2. predict 
preserve 
    * keep related x variables
    keep frsnumber time_id quarter_str hpv_status dav region orig_naics gravity compliance
	duplicates drop
	sort frsnumber time_id

	* rename, to predict 
	rename hpv_status lag_hpv_status
	rename dav lag_dav
    
	* generate lag_investment = 0 or 1
	gen lag_investment = 1
	predict fine_predicted_invest
	replace lag_investment = 0
	predict fine_predicted_noinvest

	* report conditional mean
    sum fine_predicted_invest fine_predicted_noinvest if lag_hpv_status == 0 & region == 1 & orig_naics == 21 
    sum fine_predicted_invest fine_predicted_noinvest if lag_hpv_status == 1 & region == 1 & orig_naics == 21 
	
	
	* prepare for c.
	* generate the difference 
	gen E_diff_fine = fine_predicted_noinvest - fine_predicted_invest
	* merge 
	keep frsnumber time_id E_diff_fine fine_predicted_noinvest fine_predicted_invest
	tempfile expected_fine
	save `expected_fine.dta', replace 
restore 

* prepare for c 
merge 1:1 frsnumber time_id using `expected_fine.dta'
drop _merge 

********************************************************************************
* III. reg Q3 c probit 
********************************************************************************	

probit investment E_diff_fine if compliance == 0
















