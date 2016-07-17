#include <iostream>
#include <fstream>
#include <conio.h>
#include <stdlib.h>
#include <assert.h>
using namespace std;

struct info
   {
char letter;
int number;
   };

const int NAME = 20;

int main()
   {
char outF[NAME] = "output.dat";
ofstream fout;
ifstream fin;

info test[2];
info intest[2];


test[0].letter = 'A';
test[0].number = 1;

test[1].letter = 'b';
test[1].number = 9;

cout << test[0].letter << test[0].number << " " << test[1].letter << test[1].number << endl << endl;

fout.clear();
fout.open(outF, ios:: out | ios::binary );
fout.write(reinterpret_cast<char*>(test), sizeof(test));
//fout.write(reinterpret_cast<char*>(&test2), sizeof(test2));
fout.close();

fin.clear();
fin.open(outF, ios::in | ios::binary );
fin.read(reinterpret_cast<char*>(intest), sizeof(intest));
//fin.read(reinterpret_cast<char*>(&intest2), sizeof(intest2));
fin.close();

cout << intest[0].letter << intest[0].number << " " << intest[1].letter << intest[1].number << endl << endl;

system("PAUSE");
return 0;
   }