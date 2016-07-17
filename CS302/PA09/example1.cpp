// lab10-example1.cpp
#include <iostream>
#include <cmath>
#include "HashTable.cpp"

using namespace std;

struct Account
{
    int acctNum;              // (Key) Account number
    float balance;            // Account balance

    int getKey () const { return acctNum; }
    static unsigned int hash(const int& key) { return abs( key ); }
};

int main()
{
    HashTable<Account,int> accounts(11);    // List of accounts
    Account acct;                         // A single account
    int searchKey;                        // An account key

    // Read in information on a set of accounts.

    cout << endl << "Enter account information (num balance) for 5 accounts: "
         << endl;

    for ( int i = 0; i < 5; i++ )
    {
        cin >> acct.acctNum >> acct.balance;
        accounts.insert(acct);
    }

    // Checks for accounts and prints records if found

    cout << endl;
    cout << "Enter account number (<EOF> to end): ";
    while ( cin >> searchKey )
    {
       if ( accounts.retrieve(searchKey,acct) )
           cout << acct.acctNum << " " << acct.balance << endl;
       else
           cout << "Account " << searchKey << " not found." << endl;
    }

    return 0;
}
