
********************************************************************************
* HW2
* Purpose:
* 
* Author: Jingyuan Wang
********************************************************************************

* set PATH *********************************************************************
* jingyuan's path
if c(username) == "jingyuanwang" | c(username) == "Jingyuan" {
    * mac
    if regexm(c(os),"Mac") == 1 {
        global proj_folder = "/Users/jingyuanwang/Dropbox/Course/ECON/IO/NU450/NU450_HW/450-3-HW2"
        global Git = "/Users/jingyuanwang/GitHub/NU450_HW/450-3-HW2"
    }    
    * win
    else if regexm(c(os),"Windows") == 1 {
        ***
    } 
}


global data = "$proj_folder/data"
global results = "$proj_folder/results"

********************************************************************************
* I. install packages
********************************************************************************

ssc install reghdfe, replace
