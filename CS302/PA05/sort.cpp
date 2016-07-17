#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

#include "Timer.h"

const int numSorts = 100;

void selectionSort(vector<int>::iterator front, vector<int>::iterator back) {
    for (vector<int>::iterator i = front; i != back - 1; ++i) {
	vector<int>::iterator minLoc = i;
	for (vector<int>::iterator j = i + 1; j != back; ++j) {
	    if (*minLoc > *j) {
		minLoc = j;
	    }
	}
	int temp = *i;
	*i = *minLoc;
	*minLoc = temp;
    }
}

void quickSort(vector<int>::iterator front, vector<int>::iterator back) {
    vector<int>::iterator left = front + 1;
    if (left == back) {
	return;
    }
    vector<int>::iterator right = back - 1;

    int value = *front;
    do {
	while (*left < value && left != back) ++left;
	while (*right > value && right != front) --right;

	if (left <= right) {
	    int temp = *left;
	    *left++ = *right;
	    *right-- = temp;
	}
    } while (left <= right);

    *front = *right;
    *right = value;

    if (front < right) {
	quickSort(front, right);
    }
    if (back > left) {
	quickSort(left, back);
    }
}

// This function takes a pointer to a sorting function that expects
// two iterators that point to the first and last element of the list
// to be sorted, the name of the sort routine, the list of elements to
// be sorted, and a reference to the overhead Timer object.  The routine
// makes a copy of masterList into keyList and then sorts keyList so that
// we can repeat the process a number of times with an unsorted list each
// time.
void timeSort(void (*fcn)(vector<int>::iterator front,
			  vector<int>::iterator back),
	      const string name,
	      const vector<int>& masterList,
	      const Timer& overhead)
{
    vector<int> keyList;

    Timer t;
    t.start();
    for (int i = 0; i < numSorts; ++i) {
	keyList = masterList;
	fcn(keyList.begin(), keyList.end());
    }
    t.stop();
    cout << name << " Duration: ";
    cout << t.getElapsedTime() - overhead.getElapsedTime() << endl;
}

int main(int argc, char **argv) {
    cout << endl << "Enter the number of keys: ";
    int numKeys;
    cin >> numKeys;

    // create list of values to be sorted
    vector<int> masterList;
    for (int i = 0; i < numKeys; ++i) {
	masterList.push_back(rand());
    }

    // measure the amount of time it takes to copy the master list so
    // we do not count this time into the time it takes to sort the
    // list
    vector<int> keyList;
    Timer overhead;
    overhead.start();
    for (int i = 0; i < numSorts; ++i) {
	keyList = masterList;
    }
    overhead.stop();

    // perform various sorts
    timeSort(selectionSort, "Selection sort", masterList, overhead);
    timeSort(quickSort, "Quicksort", masterList, overhead);
    timeSort(sort, "STL sort", masterList, overhead);

    return 0;
}
