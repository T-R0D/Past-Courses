//--------------------------------------------------------------------
//
//  Laboratory 10                                        test10std.cpp
//
//  Test program for the standard deviation operation in the Hash Table ADT
//
//--------------------------------------------------------------------

#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include "HashTable.cpp"

using namespace std;

struct Data
{
    public:
        void setKey ( string newKey ) { key = newKey; }
        string getKey () const { return key; }
	static unsigned int hash(const string& str)
        {
            // Uncomment each of these as you try them out.
            //-----------------------
            // Hash Algorithm 1
            //-----------------------
            // return 0;

            //-----------------------
            // Hash Algorithm 2
            //-----------------------
            // return int(str[0])*10 + str.length();
            
            //-----------------------
            // Hash Algorithm 3
            //-----------------------
            // double val = 0;
            // for (int i=0; i<str.length(); i++) 
            //     val += (val*1.1)+str[i];
            // return int (val);
            
            // Add your two hash algorithms below
            //-----------------------
            // Hash Algorithm 4
            //-----------------------
            
            //-----------------------
            // Hash Algorithm 5
            //-----------------------
            
        }
    private:
        string key;
};

int main()
{
    HashTable<Data, string> testTbl(64);

    Data testData;

    string key;

    ifstream data("std-dev.dat");

    if( ! data )
    {
        cerr << "Error opening 'std-dev.dat'" << endl;
    }
    else
    {
        while( data >> key )
        {
            testData.setKey( key );
            testTbl.insert( testData );
        }

        testTbl.showStructure();
        cout << endl << endl;
        cout << "The standard deviation is " 
             << testTbl.standardDeviation() << endl;
    }
}
