/**
    @file bayes_classifier.h

    @author Terence Henriod

    Project 1: Bayesion Minimum Error Classification

    @brief Class declarations for the BayesClassifier which can be used to
           use Bayes Minimum Average Error classifacation for two categories
           as well as specify Chernoff and Battacharyya upper error bounds.

    @version Original Code 1.00 (3/8/2014) - T. Henriod
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

#include <Eigen/Dense>  // -I /home/thenriod/Desktop/cpp_libs/Eigen_lib

using namespace std;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   GLOBAL CONSTANTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#define NUM_FEATURES 2
#define CORRECT 'o'
#define INCORRECT 'X'
#define EPSILON 0.000000000000000000000000000001L

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


struct TestData
{
  Eigen::Vector2d feature_vector;
  string actual_class;
  string classified_as;
  char   correctly_classified;
};

struct Chernoff
{
  double bound;
  double beta;
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
  void setMean( const Eigen::Vector2d& new_mean_vector, const int which_class );
  void setCovariance( const Eigen::Matrix2d& new_covariance_matrix,
                      const int which_class );
  void setPriorProbabilities( const double new_probability );
  void setAssumptionCase( const int new_case );

  /*---   Accessors   ---*/
  void performAnalysis( const string input_file, const string output_file );
  string assignToClass( Eigen::Vector2d& input_vector );
  double calculateDiscriminant( const Eigen::Vector2d& input_vector,
                                const int which_class );
  Chernoff findChernoffBound( double beta_star, int level );
  double findBattacharyyaBound();
  double kappaF( const double beta );


 protected:

 private:
  /*---   Data Members   ---*/
  double variance;
  vector<double> prior_probabilities;
  vector<Eigen::Vector2d> mean_vectors;
  vector<Eigen::Matrix2d> covariance_matrices;
  int assumption_case_;
};

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   TERMINATING PREPROCESSOR DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#endif		// #ifndef bayes_classifier.h
