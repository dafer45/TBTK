#!/bin/bash

#This file is a template for writing validation code. Validation is meant to
#ensure that data structures stay up to date with the C++ implementation in
#TBTK. Validation is only meant to ensure the that the data structure itself is
#up to date. It does therefore not ensure that the functionality is correct.
#
#

#Do not change this function.
function validate(){
	validator=TBTKValidator$1
	numTests=$($validator NumTests)
	for((n=0; n < ${numTests}; n++))
	do
		echo $($validator Get $n | $2 | $validator Validate $n)
	done
}

#Add validation code here.
#The structure of a validation command is as follows.
validate "DATA_STRUCTURE" "COMMAND"
#Here DATA_STRUCTURE is the data structure to validate

validate "Index" "python Index.py"
#The generGeneral structure
