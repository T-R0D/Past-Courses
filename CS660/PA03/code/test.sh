#! /bin/bash

#
# A script for testing the Flex tokenizer.
#

clear; clear; clear;

num_passed=0
num_failed=0

function unit_test {
	expression=$1
	expected=$2

	echo "~~~~~~~~~~~~~~~~~~~~~~~"
	echo "CASE: "
	echo "$expression"
	echo ""

	result=$(echo "$expression" | bin/calc)

	if [[ $result = "" ]]; then
		result="FAIL"
	fi

	if [[ $result -eq $expected ]]; then
		message="passed"
		((num_passed++))
	else
		message="FAILED"
		((num_failed++))
	fi

	echo "expected: ""$expected"
	echo "got:      ""$result"
	echo ""
	echo "RESULT: "$message
	echo "~~~~~~~~~~~~~~~~~~~~~~~"
}

echo "========================"
echo "| Building the project |"
echo "========================"
make

echo ""
echo "================="
echo "| Running Tests |"
echo "================="

unit_test '1;'                            1
unit_test '(2);'                          2
unit_test '1 + 2;'                        3
unit_test '7- 3;'                         4
unit_test '1 * 5;'                          5
unit_test '12 /2;'                        6
unit_test '1+2 *3;'                       7
unit_test $'4\n+\n4\n;'                   8 
unit_test '((2 + 1)* 3);'                 9
unit_test '100 / (2 * 5);'                  10
unit_test '4@+4;'                         "FAIL"
unit_test '99 / 0;'                       "FAIL"
unit_test '00 + 1;'                       "FAIL"
unit_test '1 + 01;'                       "FAIL"
unit_test '99999999999999999999999 + 1;'  "FAIL"

echo ""
echo "============================"
echo "PASSED: ""$num_passed"
echo "FAILED: ""$num_failed"
