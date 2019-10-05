cap log close
set memory 96m

use opreg.dta

xtset gvkey year

*Exit Variable
gen firmid=gvkey
sort firmid year
by firmid : gen count = _N
gen survivor = count == 8
gen has90 = 1 if year == 2002
sort firmid has90
by firmid : replace has90 = 1 if has90[_n-1] == 1
replace has90 = 0 if has90 == .
sort firmid year
by firmid : gen has_gaps = 1 if year[_n-1] != year-1 & _n != 1
sort firmid has_gaps
by firmid : replace has_gaps = 1 if has_gaps[_n-1] == 1
replace has_gaps = 0 if has_gaps == .
sort firmid year
by firmid : generate exit = survivor == 0 & has90 == 0 & has_gaps != 1 & _n == _N 
replace exit=0 if exit==1 & year==2002

gen t=year-1994
drop if missing(lninv)
xtset gvkey year
// Create terms for polynomial in (i,k,a)
gen double lninvlnkop = lninv*lnkop
gen double lninvage = lninv*age
gen double lnkopage = lnkop*age
gen double lninvsq = lninv^2
gen double lnkopsq = lnkop^2
gen double agesq = age^2


// Step I - regress lny on variable inputs and 
// polynomial in i, a, k
regress lny lnl lnm lninv lnkop age t lninvlnkop lninvage lnkopage 	///
	lninvsq lnkopsq agesq
predict double lny_hat if e(sample), xb
scalar b_lnl = _b[lnl]
scalar b_lnm = _b[lnm]
scalar b_t = _b[t]


// Step II -- Estimate probability of survival 
probit exit L.(lninv lnkop age lninvlnkop lninvage lnkopage 	///
		lninvsq lnkopsq agesq t) 
predict phat if e(sample), pr


// Step III -- Nonlinear regression of y - lnl*b_lnl - lnm*b_lnm-b_t*t
// on age, capital, and the polynomial to control for selection

// First, get phi_hat
generate double phi_hat = lny_hat - lnl*b_lnl - lnm*b_lnm-b_t*t-_b[_cons]

// Next, generate the depvar for the nonlinear equation
// Output minus the contributions of the variable inputs
generate double lhs = lny - lnl*b_lnl - lnm*b_lnm-b_t*t-_b[_cons]

// mark out missing observations
generate useme = 1
gen l1phi = L.phi_hat
gen l1lnkop = L.lnkop
gen l1age = L.age

foreach var of varlist lhs lnkop age l1phi l1lnkop l1age {
	replace useme = 0 if `var' >= .
}

gen double phat2 = phat^2

// Finally, fit the nonlinear model to get capital and age coefs.
nl ( lhs = {b0} + {bk}*lnkop + {ba}*age + 			///
	{t1}*(l1phi - {bk}*l1lnkop - {ba}*l1age) +		///
	{t1sq}*(l1phi - {bk}*l1lnkop - {ba}*l1age)^2 +		///
	{t2}*phat + {t2sq}*phat^2 +				///
	{t1t2}*(l1phi - {bk}*l1lnkop - {ba}*l1age)*phat ) 	///
	if useme
