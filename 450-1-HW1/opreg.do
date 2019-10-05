cap log close
log using opreg.log, text replace

set memory 96m

use opreg

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

xtset gvkey year

*OLS Regression

reg  lny lnl lnm lnk age t

xtset gvkey year

*Olley and Pakes (1996)

opreg lny, exit(exit) state(age lnkop) proxy(lninv) free(lnl lnm) cvars(t) vce(bootstrap, seed(1) rep(2))

lincom age + lnkop + lnl + lnm

log close

drop if lninv==.
xttab gvkey


*IV Estimates
reg  lny lnl lnm lnk age t
outreg lnl lnm lnk age t using op.doc,se 3aster ctitle("OLS") replace
ivreg2 lny lnl lnm (lnk=l2.lninv l3.lninv l4.lninv l2.lnk l3.lnk l4.lnk) age t
outreg lnl lnm lnk age t using op.doc,se 3aster ctitle("IV1") append
ivreg2 lny lnl lnm (lnk=l1.lninv l2.lninv l3.lninv l1.lnk l2.lnk l3.lnk) age t
outreg lnl lnm lnk age t using op.doc,se 3aster ctitle("IV2") append
ivreg2 lny lnl lnm (lnk=l1.lninv l2.lninv l3.lninv l1.lnl l2.lnl l3.lnl l1.lnm l2.lnm l3.lnm) age t
outreg lnl lnm lnk age t using op.doc,se 3aster ctitle("IV3") append
