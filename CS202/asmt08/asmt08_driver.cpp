#include <iostream>
#include <fstream>
#include <conio.h>
#include <stdlib.h>
#include <iomanip>
#include <assert.h>
#include <string>
#include "Ttools.h"
#include "f_filterBase.h"
#include "encryptClass.h"
#include "fileCopyClass.h"
#include "upperClass.h"
using namespace std;


int main()
   {
   // vars
   string inF   = "BADFILENAME";        // default file names
   string outFe = "a_encryptTest.txt";
   string outFu = "a_upperTest.txt";
   string outFc = "a_copyTest.txt";

   char response;

   ifstream fin;
   ofstream fout;
   encryptClass encryption;
   upperClass upperize;
   fileCopyClass copier;

   f_filterBase* pointer;

   // prompt for file names
   cout << "Enter the inptut file: ";
   cin >> inF;
   cout << "Would you like to enter names for the output files (Y or N)? "
        << endl;

   cin.ignore();  // clear stream
   response = cin.get();

   switch(response)
     {
     case 'y':
     case 'Y':
       cout << endl << "Enter the output file name for an encrypted file: ";
       cin >> outFe;
       cout << endl << "Enter the output file name for an uppercased file: ";
       cin >> outFu;
       cout << endl << "Enter the output file name for a copied file: ";
       cin >> outFc;
       cout << endl;
     break;
     }

   // test capabilities

   // encryption
   pointer = &encryption;
   pointer->set_inF(inF);
   pointer->set_outF(outFe);
   encryption.set_key(7);    // pointers to base classes can't access derived class
                             // member functions
   pointer->doFilter(fin, fout);

   // uppercasing
   upperize.set_inF(inF);
   upperize.set_outF(outFu);
   upperize.doFilter(fin, fout);

   // copying
   copier.set_inF(inF);
   copier.set_outF(outFc);
   copier.doFilter(fin, fout);

   holdProg();

   // return 0
   return 0;
   };