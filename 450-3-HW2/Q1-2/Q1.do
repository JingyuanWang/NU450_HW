********************************************************************************
* HW2
* Purpose: Q1
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
* III. Reg Q1a
* also tried with year FE or quarterFE, no difference
* significant if include ordered_violated
********************************************************************************
*reghdfe investment hpv_status dav , a( i.region i.orig_naics i.gravity) cluster(orig_naics) 

quietly{
capture eststo clear
eststo: reghdfe investment hpv_status dav, a( i.region i.orig_naics i.gravity) cluster(orig_naics) 
estadd local gravityFE "Yes"
estadd local regionFE "Yes"
estadd local naicsFE "Yes"

eststo: reghdfe investment hpv_status dav lag_investment, a( i.region i.orig_naics i.gravity) cluster(orig_naics) 
estadd local gravityFE "Yes"
estadd local regionFE "Yes"
estadd local naicsFE "Yes"

eststo: reghdfe investment hpv_status dav violation fine inspection, a( i.region i.orig_naics i.gravity) cluster(orig_naics) 
estadd local gravityFE "Yes"
estadd local regionFE "Yes"
estadd local naicsFE "Yes"

eststo: reghdfe investment hpv_status dav , a( i.region i.orig_naics i.gravity i.time) cluster(orig_naics) 
estadd local gravityFE "Yes"
estadd local regionFE "Yes"
estadd local naicsFE "Yes"
estadd local timeFE "Yes"

}
estout,  ///
        cells(b(star fmt(3)) se(par fmt(3)))  ///
        modelwidth(10) varwidth(10) /// 
		mlabels("investment" "investment" "investment" "investment" ) ///
		stats(r2_a N gravityFE naicsFE regionFE timeFE, fmt(%9.3f %9.0g) labels(R-squared)) ///
		drop( _cons )
	
estout using "$results/Q1a.tex" , style(tex) ///
        cells(b(star fmt(3)) se(par fmt(3)))  ///
        modelwidth(10) varwidth(10) /// 
		mlabels("investment" "investment" "investment" "investment" ) ///
		stats(r2_a N gravityFE naicsFE regionFE timeFE, fmt(%9.3f %9.0g) labels(R-squared)) ///
		drop( _cons ) ///
		varlabels( hpv_status "HPV" dav "DAV"  ) ///
		replace
		
********************************************************************************
* III. Reg Q1b
********************************************************************************
 
***************************************
* construct independent variables
***************************************
gen lag_inspection = L.inspection
gen lag_fine = L.fine 
gen lag_violation = L.violation

bysort frsnumber (time_id): egen violation_tot = total(violation)
bysort frsnumber (time_id): egen fine_tot = total(fine)
bysort frsnumber (time_id): egen inspection_tot = total(inspection)

bysort frsnumber (time_id): gen violation_cumul = sum(violation)
bysort frsnumber (time_id): gen fine_cumul = sum(fine)
bysort frsnumber (time_id): gen inspection_cumul = sum(inspection)

***************************************
* collapse to region-naics-gravity-quarter level
***************************************
preserve
collapse (mean) compliance lag_compliance lag_hpv_status lag_violator_nothpv dav *_cumul *_tot ///
              investment lag_investment lag2_investment ///
              inspection fine violation lag_inspection lag_fine lag_violation ///
			  , by(omega1 region naics_recode orig_naics gravity year quarter time_id)
egen region_ind_code = group(region orig_naics)

bysort omega1 (time_id): gen cumul_inspection = sum(inspection)
bysort omega1 (time_id): gen cumul_fine = sum(fine)
bysort omega1 (time_id): gen cumul_violation = sum(violation)
bysort omega1 (time_id): gen n = _n 
foreach var in cumul_inspection cumul_fine cumul_violation{
	replace `var' = `var'/n
}

xtset omega1 time_id
***************************************
* reg
***************************************
capture eststo clear
quietly{

eststo: reghdfe compliance inspection fine violation , a(region orig_naics gravity time_id) cluster(region_ind_code)
estadd local gravityFE "Yes"
estadd local regionFE "Yes"
estadd local naicsFE "Yes"
estadd local timeFE "Yes"

eststo: reghdfe compliance lag_inspection lag_fine lag_violation, a(region orig_naics gravity time_id) cluster(region_ind_code)
estadd local gravityFE "Yes"
estadd local regionFE "Yes"
estadd local naicsFE "Yes"
estadd local timeFE "Yes"

eststo: reghdfe compliance inspection fine violation lag_inspection lag_fine lag_violation, a(region orig_naics gravity time_id) cluster(region_ind_code)
estadd local gravityFE "Yes"
estadd local regionFE "Yes"
estadd local naicsFE "Yes"
estadd local timeFE "Yes"

eststo: reghdfe compliance inspection fine violation lag_inspection lag_fine lag_violation lag_hpv_status lag_violator_nothpv dav  , a(region orig_naics gravity time_id) cluster(region_ind_code)
estadd local gravityFE "Yes"
estadd local regionFE "Yes"
estadd local naicsFE "Yes"
estadd local timeFE "Yes"


}
estout,  ///
        cells(b(star fmt(3)) se(par fmt(3)))  ///
        modelwidth(10) varwidth(10) /// 
		mlabels("compliance" "compliance" "compliance" "compliance" "compliance") ///
		stats(r2_a N gravityFE  naicsFE regionFE timeFE, fmt(%9.3f %9.0g) labels(R-squared)) ///
		drop( _cons )
		

estout using "$results/Q1b.tex" , style(tex) ///
        cells(b(star fmt(3)) se(par fmt(3)))  ///
        modelwidth(10) varwidth(10) /// 
		mlabels("compliance" "compliance" "compliance" "compliance" "compliance") ///
		stats(r2_a N gravityFE naicsFE regionFE timeFE, fmt(%9.3f %9.0g) labels(R-squared)) ///
		drop( _cons ) ///
		replace
		
restore 

		
********************************************************************************
* DONE
********************************************************************************		

		
		
		
		
