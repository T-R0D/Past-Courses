#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

#include "Timer.h"

const int numSearches = 100000;

// ABC that gives an interface for the various search implementations
// This allows each class to act like a function by defining
// operator() which takes a search value and sorted list of keys to
// look through.  We do this so that we can use STL functions (e.g.,
// for_each) to // iterate through a list of keys to try since they
// expect to call a function for each element.
class Search : public binary_function<int, vector<int>, bool> {
    virtual bool operator()(int searchValue, const vector<int>& keys) const = 0;
};

class linearSearch : public Search {
  public:
    bool operator()(int searchValue, const vector<int>& keys) const {
	for (int i = 0; i < keys.size(); ++i) {
	    if (keys[i] == searchValue) {
		return true;
	    }
	}

	return false;
    }
};

class binarySearch : public Search {
  public:
    bool operator()(int searchValue, const vector<int>& keys) const {
	int low = 0;
	int high = keys.size() - 1;

	while (low <= high) {
	    int index = (low + high) / 2;
	    int key = keys[index];
	    if (searchValue < key) {
		high = index - 1;
	    } else if (searchValue > key) {
		low = index + 1;
	    } else {
		return true;
	    }
	}

	return false;
    }
};

class STLSearch : public Search {
  public:
    bool operator()(int searchValue, const vector<int>& keys) const {
	vector<int>::const_iterator loc = find(keys.begin(),
					       keys.end(),
					       searchValue);

	return loc != keys.end();
    }
};

int main(int argc, char **argv) {
    cout << endl << "Enter the number of keys: ";
    int numKeys;
    cin >> numKeys;

    // create list of values to search through
    vector<int> keyList;
    for (int i = 0; i < numKeys; ++i) {
	keyList.push_back(rand());
    }

    sort(keyList.begin(), keyList.end());

    // create list of values to search for
    vector<int> searchKeys;
    for (int i = 0; i < numSearches; ++i) {
	searchKeys.push_back(rand());
    }

    // time linear search
    Timer t;
    t.start();
    for_each(searchKeys.begin(),
	     searchKeys.end(),
	     bind2nd(linearSearch(), keyList) );
    t.stop();
    cout << "Linear Duration: " << t.getElapsedTime() << endl;

    // time binary search
    Timer u;
    u.start();
    for_each(searchKeys.begin(),
	     searchKeys.end(),
	     bind2nd(binarySearch(), keyList) );
    u.stop();
    cout << "Binary Duration: " << u.getElapsedTime() << endl;

    // time STL search
    Timer v;
    v.start();
    for_each(searchKeys.begin(),
	     searchKeys.end(),
	     bind2nd(STLSearch(), keyList) );
    v.stop();
    cout << "STL Duration: " << v.getElapsedTime() << endl;

    return 0;
}
