
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

cd "$proj_folder/"


* Install Packages *************************************************************
net install st0145_2, force
net install st0060
net install st0460
net install sg71

