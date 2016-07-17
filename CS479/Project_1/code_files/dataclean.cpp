#include <fstream>
#include <vector>
using namespace std;

int main()
{
  fstream file;
  int ndx = 0;
  double feature1;
  double feature2;
  vector<double> feature1data(10000);
  vector<double> feature2data(10000);

  ndx = 0;
  file.clear();
  file.open( "BebisPart1Class1.txt", fstream::in );
  while( file.good() )
  {
    file >> feature1data[ndx] >> feature2data[ndx];
    ndx++;
  }
  file.close();

  file.clear();
  file.open( "Bebis11.txt", fstream::out );
  for( ndx = 0; ndx < feature1data.size(); ndx++ )
  {
    file << feature1data[ndx] << ", " << feature2data[ndx] << ", ONE\n";
  }
  file.close();
  

  ndx = 0;
  file.clear();
  file.open( "BebisPart1Class2.txt", fstream::in );
  while( file.good() )
  {
    file >> feature1data[ndx] >> feature2data[ndx];
    ndx++;
  }
  file.close();

  file.clear();
  file.open( "Bebis12.txt", fstream::out );
  for( ndx = 0; ndx < feature1data.size(); ndx++ )
  {
    file << feature1data[ndx] << ", " << feature2data[ndx] << ", TWO\n";
  }
  file.close();





  ndx = 0;
  file.open( "BebisPart2Class1.txt", fstream::in );
  while( file.good() )
  {
    file >> feature1data[ndx] >> feature2data[ndx];
    ndx++;
  }
  file.close();

  file.clear();
  file.open( "Bebis21.txt", fstream::out );
  for( ndx = 0; ndx < feature1data.size(); ndx++ )
  {
    file << feature1data[ndx] << ", " << feature2data[ndx] << ", ONE\n";
  }
  file.close();
  

  ndx = 0;
  file.clear();
  file.open( "BebisPart2Class2.txt", fstream::in );
  while( file.good() )
  {
    file >> feature1data[ndx] >> feature2data[ndx];
    ndx++;
  }
  file.close();

  file.clear();
  file.open( "Bebis22.txt", fstream::out );
  for( ndx = 0; ndx < feature1data.size(); ndx++ )
  {
    file << feature1data[ndx] << ", " << feature2data[ndx] << ", TWO\n";
  }
  file.close();


  return 0;
}
