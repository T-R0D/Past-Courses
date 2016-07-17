/**
    @file ProgramTest.cpp

    @author Terence Henriod

    Programming Exercise 1: Login Authentication

    @brief This program tests the HashTable ADT by using it for login
           authentication purposed with a sample data file.

    @version Original Code 1.00 (11/2/2013) - T. Henriod
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <iostream>
#include <fstream>
#include <string>
#include "HashTable.cpp"
using namespace std;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   GLOBAL CONSTANTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

const int HASH_SIZE = 8; // assume a small database
const char* FILE_NAME = "password.dat";  // the file to be used in this exercise


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   GLOBAL FUNCTION PROTOTYPES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// none


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   STRUCT / CLASS DEFINITIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
@class Credentials

This struct will contain a user name and their password. To be compatible with
the HashTable ADT, this struct supports hash() and getKey() functions.
Hashing occurs by multiplying the ASCCI values of the first two characters in a
string.
*/
class Credentials
{
 public:
  /*~~~     METHODS     ~~~*/

  /**
  getKey

  Returns the userName member string as a key for this object.

  @return userName   The username currently held by the object.
  */
  string getKey() const
  { return userName; };


  /**
  hash

  Hashes the username for compatibility with the HashTable ADT.
  Uses the hashing function used in the lab manual package
  provided in test10.cpp

  @return hashResult   The result of the hashing function.

  @detail @bAlgorithm
  -# The first two characters of the string are multiplied to
      produce the hash key.
  */
  static unsigned int hash( const string& name )
  {
  // variables
  unsigned int hashResult = 0;

  // sum the ASCII values of the string
  for (int i = 0; i < name.length(); ++i)
  {
	hashResult += name[i];
  }

  // return the hash value
  return hashResult;
  };

  /*~~~     DATA MEMBERS     ~~~*/
  string userName;   // the user's username
  string password;   // the user's password
};

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   MAIN FUNCTION DEFINITION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
main

The driving function for the program. Reads in login credentials from the
specified file, then allows a user to attemp to "login" by entering username
and password combinations. If the "login" is successful, then the program
reports it, otherwise, the program reports failure. This continues until the
EOF flag is reached in the console input stream.

@return 0
*/
int main()
{
  // variable(s)
  Credentials temp;
  ifstream credentialFile;
  HashTable<Credentials, string> userData( HASH_SIZE );
  string userName;
  string password;
  bool loginSuccess = false;

  // clear file object and open the login data file
  credentialFile.clear();
  credentialFile.open( FILE_NAME );

  // case: file opening was successful
  if( credentialFile.good() )
  {
    // prime the loop
    credentialFile >> temp.userName;
    credentialFile >> temp.password;
 
    // read in data while possible
    while( credentialFile.good() )
    {
      // add the data to the hash table
      userData.insert( temp );

      // read in the next set of data
      credentialFile >> temp.userName;
      credentialFile >> temp.password;
    }

HashTable<Credentials, string> otherGuy( userData );
otherGuy.showStructure();

    // while the end of the input data is not reached:
    cout << "Login: ";
    while( cin >> userName )
    {
      // login was not successful yet
      loginSuccess = false;

      // prompt the user for their password
      cout << "Password: ";
      cin >> password;

      // attempt to authenticate
      // case: authentication was successful
      if( otherGuy.retrieve( userName, temp ) )
      {
        // case: the password is a match
        if( password == temp.password )
        {
          // report that authentication was successful
          loginSuccess = true;
          cout << "Authentication successful" << endl;
        }
      }

      // case: authentication was not successful
      if( !loginSuccess )
      {
      cout << "Authentication failure" << endl;
      }

    cout << "Login: ";
    }
  }

  // return 0 upon completion
  return 0;
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   FUNCTION IMPLEMENTATIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// none

