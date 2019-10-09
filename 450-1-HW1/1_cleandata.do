

********************************************************************************
* 1. import and clean the data file + save as PS1_data_cleaned.dta
********************************************************************************

* 1. import file
use "$data/PS1_data.dta", clear

* 2. rename variables
{
rename X03 ly
rename X04 s01
rename X05 s02
rename X06 s03
rename X07 s04
rename X08 s05
rename X09 s06
rename X10 s07
rename X11 s08
rename X12 s09
rename X13 s10
rename X14 s11
rename X15 s12
rename X16 s13
rename X17 s14
rename X18 s15
rename X19 s16
rename X20 s17
rename X21 s18
rename X22 t4
rename X23 t6
rename X24 d90
rename X25 d91
rename X26 d92
rename X27 d93
rename X28 d94
rename X29 d95
rename X30 d96
rename X31 d97
rename X32 d98
rename X33 d99
rename X34 merger
rename X35 sciss
rename X36 lrd
rename X37 pri
rename X38 pdi
rename X39 linv
rename X40 lc
rename X41 lno
rename X42 lh
rename X43 ll
rename X44 lm
rename X45 lp
rename X46 lpci
drop X47
rename X48 lwl
rename X49 lwm
rename X50 ptw
rename X51 pwc
rename X52 peg
rename X53 ptc
rename X54 ts
rename X55 md
rename X56 inc
rename X57 ioc
rename X58 age
drop X59
rename X60 ent
rename X61 exit
drop sequence
}


* 3. industry id
gen industry_id = s01*1+s02*2+s03*3+s04*4+s05*5+s06*6+s07*7+s08*8+s09*9+s10*10+s11*11+s12*12+s13*13+s14*14+s15*15+s16*16+s17*17+s18*18

* 4. output var
gen ly_va = log(exp(ly)-exp(lm-lp+lwm))


* 5. save
order industry_id firm_id year obs ly ly_va 
order s* t4 t6 d*, a(exit)
sort industry_id firm_id year
save "$data/PS1_data_cleaned.dta", replace









