
********************************************************************************
* 450-1-HW1
* Purpose:
* 
* Author: Jingyuan Wang
********************************************************************************

* set PATH *********************************************************************
* jingyuan's path
if c(username) == "jingyuanwang" | c(username) == "Jingyuan" {
    * mac
    if regexm(c(os),"Mac") == 1 {
        global proj_folder = "/Users/jingyuanwang/Dropbox/Course/ECON/IO/NU450/NU450_HW/450-1-HW1"
        global Git = "/Users/jingyuanwang/GitHub/NU450_HW/450-1-HW1"
    }    
    * win
    else if regexm(c(os),"Windows") == 1 {
        ***
    } 
}


global data = "$proj_folder/data"
global results = "$proj_folder/results"

********************************************************************************
* 0. import and clean the data file
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




********************************************************************************
* I. statistics 
********************************************************************************
