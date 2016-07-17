
#include <stdexcept>
#include <iostream>

using namespace std;

#include <vector>

class TestVector {
  public:
    TestVector(int size);
    TestVector(const TestVector& rhs);

    TestVector& operator++();
    TestVector operator++(int ignored);

    int operator[](int loc) const;

  private:
    vector<int> values;
};
