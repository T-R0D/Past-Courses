/**
    @file database.cpp

    @author Terence Henriod

    Laboratory 11, In-lab exercise 1

    @brief (Shell) Indexed accounts database program. This program should be
           used with the database file accounts.dat.

    @version Original Code 1.00 (10/29/2013) - Modified by T. Henriod
*/
//--------------------------------------------------------------------
//
//  Laboratory 11, In-lab Exercise 1                     database.cs
//
//  (Shell) Indexed accounts database program
//
//--------------------------------------------------------------------

// Builds a binary search tree index for the account records in the
// text file accounts.dat.

#include <iostream>
#include <fstream>
#include "BSTree.cpp"

using namespace std;

//--------------------------------------------------------------------
//
// Declarations specifying the accounts database
//

const int nameLength      = 11;   // Maximum number of characters in
                                  //   a name
const long bytesPerRecord = 37;   // Number of bytes used to store
                                  //   each record in the accounts
                                  //   database file


/**
@struct AccountRecord

The struct that has fields for all data types in a database record entry.
*/
struct AccountRecord
{
    int acctID;                   // Account identifier
    char firstName[nameLength],   // Name of account holder
         lastName[nameLength];
    double balance;               // Account balance
};

//--------------------------------------------------------------------
//
// Declaration specifying the database index
//

/**
@struct IndexEntry

The struct that pairs the account ID number and the record number that
is used by the program
*/
struct IndexEntry
{
    int acctID;              // (Key) Account identifier
    long recNum;             // Record number


/**
getKey

Returns the key of the IndexEntry struct. This is the account number
for a database entry

@pre
-# the struct should contain valid data

@post
-# an account number corresponding to a record number is returned

@code
@endcode
*/
    int getKey() const
        { return acctID; }   // Return key field
};

//--------------------------------------------------------------------

int main ()
{
    ifstream acctFile( "accounts.dat" );   // Accounts database file
    AccountRecord acctRec;                // Account record
    BSTree<IndexEntry,int> index;         // Database index
    IndexEntry entry;                     // Index entry
    int searchID;                         // User input account ID
    long recNum = 0;                          // Record number

    // Iterate through the database records. For each record, read the
    // account ID and add the (account ID, record number) pair to the
    // index.

    // prime loop
    acctFile >> entry.acctID;

    // read in all record numbers
    while( acctFile.good() )
    {
      // assign the account number an index (record number)
      entry.recNum = recNum;

      // store the index entry in the tree
      index.insert( entry );

      // move on to the next record number
      recNum++;

      // move to appropriate place in file
      acctFile.seekg( ( recNum * bytesPerRecord ), acctFile.beg );

      // attempt to read next account number
      acctFile >> entry.acctID;
    }

    // Output the account IDs in ascending order.
    cout << endl << "Account IDs :" << endl;
    index.writeKeys();
    cout << endl;

    // Clear the status flags for the database file.
    acctFile.clear();
    acctFile.seekg( 0 );

    // Read an account ID from the keyboard and output the
    // corresponding record.
    // prompt for the account ID number
    cout << "Enter account ID : ";

    // read in user input and display output until the user quits
    while( cin >> searchID )
    {

      // attempt data item retrival
      // case: retrieval was successful
      if( index.retrieve( searchID, entry ) )
      {
      acctFile.seekg( ( entry.recNum * bytesPerRecord ), acctFile.beg );
      acctFile >> acctRec.acctID >> acctRec.firstName >> acctRec.lastName
               >> acctRec.balance;
      cout << entry.recNum << " : " << acctRec.acctID << ' ' << acctRec.firstName << ' '
           << acctRec.lastName << ' ' << acctRec.balance << endl;
      }
      // case: retrieval failed
      else
      {
        cout << "No record with that account ID" << endl;
      }

      // display second prompt
      cout << "Enter account ID (EOF to quit): ";
    }

  // close the file
  acctFile.close();

  // terminate program
  return 0;
}
