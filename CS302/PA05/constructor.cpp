#include <iostream>
#include <string>

using namespace std;

#include "Timer.h"
#include "TestVector.h"

const int numRepetitions = 1000000;

// function to perform computation on an array data type (e.g., vector and
// TestVector)
template <typename DataType>
int testCompute(DataType value) {
    return value[0];
}

// specialized computation function for ints
template <>
int testCompute<int>(int value) {
    return value;
}

// specialized computation function for doubles
template <>
int testCompute<double>(double value) {
    return int(value);
}

// function that tests the cost of putting a constructor inside a loop
// versus putting the constructor immediately preceding the loop
template <typename DataType>
void testConstructor(int numValues, string name) {
    DataType test(numValues);

    int junk;

    // test for constructor before loop
    Timer t;
    t.start();
    DataType temp(test);
    for (int i = 0; i < numRepetitions; ++i) {
	junk += testCompute(temp);
    }
    t.stop();
    cout << "Cost for outside constructor (" << name << "): ";
    cout << t.getElapsedTime() << endl;

    // test for constructor inside loop
    Timer u;
    u.start();
    for (int i = 0; i < numRepetitions; ++i) {
	DataType temp(test);
	junk += testCompute(temp);
    }
    u.stop();
    cout << "Cost for inside constructor (" << name << "): ";
    cout << u.getElapsedTime() << endl;
}

// nice C preprocessor hack that calls the testConstructor function
// passing the numValues variable and a string indicating the type
// passed to testConstructor
#define runTest(Type) testConstructor<Type>(numValues, #Type)

int main(int argc, char **argv) {
    cout << endl << "Enter the number of values to test: ";
    int numValues;
    cin >> numValues;

    runTest(int);
    runTest(double);
    runTest(vector<int> );
    runTest(TestVector);

    return 0;
}
