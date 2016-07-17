#include <functional>
#include <algorithm>

using namespace std;

#include "TestVector.h"

TestVector::TestVector(int size) : values(size, 0) {
}

TestVector::TestVector(const TestVector& rhs) : values(rhs.values) {
}


// pre
TestVector& TestVector::operator++() {
    transform(values.begin(),
	      values.end(),
	      values.begin(),
	      bind2nd(plus<int>(), 1) );

    return *this;
}


// post
TestVector TestVector::operator++(int ignored) {
    TestVector temp(*this);
    operator++();
    return temp;
}

int TestVector::operator[](int loc) const {
    return values[loc];
}
