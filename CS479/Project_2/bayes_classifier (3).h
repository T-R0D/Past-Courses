/**
    @file .h

    @author Terence Henriod

    Project Name

    @brief Class declarations for...


    @version Original Code 1.00 (10/29/2013) - T. Henriod
*/


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   PREPROCESSOR DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#ifndef ___BAYES_CLASSIFIER_H___
#define ___BAYES_CLASSIFIER_H___


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
using namespace std;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   GLOBAL CONSTANTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#define NUM_FEATURES 2

struct FeatureVector
{
  vector<double> data;
};

struct Matrix
{
  vector<vector<double> > data;
};

enum ClassNumber
{
  CLASS_ONE = 0,
  CLASS_TWO
};

enum AssumptionType
{
  CASE_ONE,
  CASE_TWO,
  CASE_THREE
};

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
                   CLASS DEFINITION(S)
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
@class BayesClassifier

dfsa;jlkdfs;ljkfdsa
@var lsakjdlfdsj
*/
class BayesClassifier
{
 public:
  /*---   Constructor(s) / Destructor   ---*/
  BayesClassifier();
  BayesClassifier( const BayesClassifier& other );
  BayesClassifier& operator=( const BayesClassifier& other );
  ~BayesClassifier();

  /*---   Mutators   ---*/
  void clear();
  void setMean( const double val_one, const double val_two, const int which_class );
  void setCovariance( const double covar_0_0, const double covar_0_1,
                      const double covar_1_0, double covar_1_1,
                      const int which_class );
  void setPriorProbability( const double probability, const int which_class );
  void reevaluateAssumptions();

  /*---   Accessors   ---*/
  

  /*---   Tools   ---*/
  string& assignToClass( const vector<double>& input_vector );

  double calculateDiscriminant( const vector<double>& input_vector, const int which_class );

  double findDeterminant( const vector<vector<double> >& matrix );

  vector<vector<double> >& constantTimesMatrix(
      const double constant,
      const vector<vector<double> > matrix );

  vector<vector<double> >& matrixInverse( const vector<vector<double> >& matrix );

  vector<double>& matrixTimesColumn( const vector<vector<double> >& matrix,
                                   const vector<double>& column );

  vector<double>& rowTimesMatrix( const vector<double>& row,
                                const vector<vector<double> >& matrix );

  vector<double>& constantTimesVector( const double constant, const vector<double>& the_vector );

  double rowTimesColumn( const vector<double>& row, const vector<double>& column );

  double firstSumTerm( const vector<double>& input_vector, const int which_class );

  double secondSumTerm( const vector<double>& input_vector, const int which_class );

  double thirdSumTerm( const int which_class );

  double fourthSumTerm( const int which_class );

  double fifthSumTerm( const int which_class );


  double findChernoffBound( const double  beta );
  double findBhattacharyyaBound();

 private:
  /*---   Data Members   ---*/
  vector<vector<double> > means;
  vector<double> prior_probabilities;
  vector<vector<double> > covariance_matrix_one;
  vector<vector<double> > covariance_matrix_two;
  unsigned int assumptions_case;
};

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   TERMINATING PREPROCESSOR DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#endif		// #ifndef bayes_classifier.h


