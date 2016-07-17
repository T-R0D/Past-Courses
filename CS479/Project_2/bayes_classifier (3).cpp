
/**
    @file bayes_classifier.cpp

    @author Terence Henriod

    Project Name

    @brief Class implementations declarations for...

    @version Original Code 1.00 (10/29/2013) - T. Henriod
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// Class Declaration
#include "bayes_classifier.h"

// Other Dependencies
#include <cassert>
#include <iostream>
#include <fstream>


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
                   CLASS FUNCTION IMPLEMENTATIONS
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   CONSTRUCTOR(S) / DESTRUCTOR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
GameState

The default constructor for a game state. Constructs and initilizes an empty
GameState.

@pre
-# The GameState object is given an appropriate identifier.

@post
-# A new, empty GameState will be initialized.

@code
@endcode
*/
BayesClassifier::BayesClassifier()
{
  vector<double> temp(NUM_FEATURES);

  means.push_back( temp );
  means.push_back( temp );

  prior_probabilities = temp;

  covariance_matrix_one = means;
  covariance_matrix_two = means;
  assumptions_case = CASE_THREE;

  // no return - constructor
}


BayesClassifier::BayesClassifier( const BayesClassifier& other )
{
  *this = other;

  // no return - constructor
}

BayesClassifier& BayesClassifier::operator=( const BayesClassifier& other )
{

  if( this != &other )
  {
    means = other.means;

    prior_probabilities = other.prior_probabilities;

    covariance_matrix_one = other.covariance_matrix_one;
    covariance_matrix_two = other.covariance_matrix_two;

    assumptions_case = other.assumptions_case;
  }

  return *this;
}

BayesClassifier::~BayesClassifier()
{
  // currently nothing to destruct
}

  /*---   Mutators   ---*/
void BayesClassifier::clear()
{
  // currently not clearing TODO: implement
}

void BayesClassifier::setMean( const double val_one, const double val_two, const int which_class )
{
  means[which_class][0] = val_one;
  means[which_class][1] = val_two;
}

void BayesClassifier::setCovariance( const double covar_0_0, const double covar_0_1,
                      const double covar_1_0,double covar_1_1,
                      const int which_class )
{
  if( which_class == 0 )
  {
    covariance_matrix_one[0][0] = covar_0_0;
    covariance_matrix_one[0][1] = covar_0_1;
    covariance_matrix_one[1][0] = covar_1_0;
    covariance_matrix_one[1][1] = covar_1_1;
  }
  else
  {
    covariance_matrix_two[0][0] = covar_0_0;
    covariance_matrix_two[0][1] = covar_0_1;
    covariance_matrix_two[1][0] = covar_1_0;
    covariance_matrix_two[1][1] = covar_1_1;
  }
}


void BayesClassifier::setPriorProbability( const double probability, const int which_class )
{
  prior_probabilities[which_class] = probability;
}


void BayesClassifier::reevaluateAssumptions()
{
  // TODO: implement!
}

  /*---   Accessors   ---*/
  

  /*---   Tools   ---*/
string& BayesClassifier::assignToClass( const vector<double>& input_vector )
{
  string result = "Class One";
  double discriminant_one = 0;
  double discriminant_two = 0;
  double difference = 0;

  discriminant_one = calculateDiscriminant( input_vector, 0 );
  discriminant_two = calculateDiscriminant( input_vector, 1 );

  difference = discriminant_one - discriminant_two;

  if( difference <= 0 )
  {  
    result = "Class Two";
  }

  return result;
}


double BayesClassifier::calculateDiscriminant( const vector<double>& input_vector,
                                               const int which_class )
{

// g(x) = -1/2 (x - u)E^{-1} - d/2 * ln(2pi) - 1/2 * ln|E| + ln(P(w))

  // variables
  double result = 0;
  double first_term = 0;
  double second_term = 0;
  double third_term = 0;
  double fourth_term = 0;
  double fifth_term = 0;

  // case: we are using case 1 assumptions (covariance matrices are
  //       identical and arbitrary)
  if( assumptions_case == CASE_ONE )
  {
// \frac{(x - u)^{t} * (x - u)}{2s^2} + ln(P(w))
  }
  // case: the covariance matrices are identical but non-diagonal
  //       (case 2) 
  else if( assumptions_case == CASE_TWO )
  {

  }
  // case: we are making no assumptions
  else
  {
// g(x) = x^t * (-1/2 * E^-1) * x + (E^-1 * u)^t * x + -1/2 * u^t * E^-1 * u - 1/2 * ln |E| + ln P(w)

    // perform the x^t * (-1/2 * E^-1) * x part of the computation
    first_term = firstSumTerm( input_vector, which_class );

    // (E^-1 * u)^t * x part
    first_term = firstSumTerm( input_vector, which_class );

    // -1/2 * u^t * E^-1 * u part
    first_term = firstSumTerm( input_vector, which_class );

    // 1/2 * ln |E| part
    first_term = firstSumTerm( input_vector, which_class );

    // ln P(w) part
    first_term = firstSumTerm( input_vector, which_class );

    // add it all up
    result = first_term + second_term + third_term + fourth_term + fifth_term;
  }

  // return the result
  return result; 
}


double BayesClassifier::firstSumTerm( const vector<double>& input_vector, const int which_class )
{
// x^t * (-1/2 * E^-1) * x
  // variables
  double first_sum_term = 0;
  vector<vector<double>> covariance_term;
  vector<double> intermediate_row( NUM_FEATURES );

  // find the middle multiplicative term
  if( which_class == 0 )
  {
    covariance_term = covariance_matrix_one;
  }
  else
  {
    covariance_term = covariance_matrix_two;
  }
  covariance_term = matrixInverse( covariance_term );
  covariance_term = constantTimesMatrix( -0.5, covariance_term );

  // perform the first matrix multiplication
  intermediate_row = rowTimesMatrix( input_vector, covariance_term );

  // perform the second matrix multiplication
  first_sum_term = rowTimesColumn( intermediate_row, input_vector );

  // return the result
  return first_sum_term;
}


double BayesClassifier::secondSumTerm( const vector<double>& input_vector,
                                       const int which_class )
{
// (E^-1 * u)^t * x part
  // variables
  double second_sum_term = 0;
  vector<vector<double>> covariance_term;
  vector<double> intermediate_col_then_row(NUM_FEATURES);

  // get the inverted covariance matrix
  if( which_class == 0 )
  {
    covariance_term = covariance_matrix_one;
  }
  else
  {
    covariance_term = covariance_matrix_two;
  }
  covariance_term = matrixInverse( covariance_term );

  // perform the parenthesised multiplication
  intermediate_col_then_row = matrixTimesColumn( covariance_term, means[which_class] );

  // perform the final multiplication
  second_sum_term = rowTimesColumn( intermediate_col_then_row, input_vector );

  // return the result
  return second_sum_term;
}


double BayesClassifier::thirdSumTerm( const int which_class )
{
// -1/2 * u^t * E^-1 * u part
  // variables
  double third_sum_term = 0;
  vector<vector<double>> covariance_term;
  vector<double> intermediate_vector(NUM_FEATURES);

  // setup the intermediate vector
  intermediate_vector = constantTimesVector( -0.5, means[which_class] );

  // get the inverted matrix
  if( which_class == 0 )
  {
    covariance_term = covariance_matrix_one;
  }
  else
  {
    covariance_term = covariance_matrix_two;
  }
  covariance_term = matrixInverse( covariance_term );

  // perform the first matrix multiplication
  intermediate_vector = rowTimesMatrix( intermediate_vector, covariance_term );

  // perform the other matrix multiplication
  third_sum_term = rowTimesColumn( intermediate_vector, means[which_class] );

  // return the result
  return third_sum_term;
}

double BayesClassifier::fourthSumTerm( const int which_class )
{
// 1/2 * ln |E| part
  // variables
  double fourth_sum_term = 0;
  double log_argument = 0;

  // case: we are using the first class
  if( which_class == 0 )
  {
    log_argument = findDeterminant( covariance_matrix_one );
  }
  else
  {
    log_argument = findDeterminant( covariance_matrix_two );
  }

  // compute the rest of the expression
  fourth_sum_term = 0.5 * log( log_argument );

  return fourth_sum_term;
}

double BayesClassifier::fifthSumTerm( const int which_class )
{
  // simply return the prior probability of the specified class
   return prior_probabilities[which_class];
}


double BayesClassifier::findDeterminant( const vector<vector<double>>& matrix )
{
  // variables
  double result;
  double a_times_d = 0;
  double b_times_c = 0;

  // ad - bc
  // find ad
  a_times_d = matrix[0][0] * matrix[1][1];

  // find bc
  b_times_c = matrix[0][1] * matrix[1][0];

  // take the difference
  result = a_times_d - b_times_c;

  // return the result
  return result;
}


vector<vector<double>>& BayesClassifier::constantTimesMatrix(
    const double constant,
    const vector<vector<double>> matrix )
{
  // variables
  vector<vector<double>> new_matrix = matrix;
  int row = 0;
  int col = 0;

  // visit every row of the matrix
  for( row = 0; row < NUM_FEATURES; row++ )
  {
    // visit every element
    for( col = 0; col < NUM_FEATURES; col++ )
    {
      // compute and store the new value
      new_matrix[row][col] = constant * matrix[row][col];
    }
  }

  // return the resulting matrix
  return new_matrix;
}


vector<vector<double>>& BayesClassifier::matrixInverse( const vector<vector<double>>& matrix )
{
  // variables
  vector<vector<double>> inverted_matrix = matrix;
  double determinant_inverse = 0;
  
  // 1/(ad - bc) * matrix
  // find the inverse of the determinant
  determinant_inverse = 1.0 / findDeterminant( matrix );

  // multiply the determinant by the original matrix
  inverted_matrix = constantTimesMatrix( determinant_inverse, matrix );

  // return the newly inverted matrix
  return inverted_matrix;
}


vector<double>& BayesClassifier::matrixTimesColumn( const vector<vector<double>>& matrix,
                                   const vector<double>& column )
{
  // variables
  vector<double> result_column(NUM_FEATURES);
  int row = 0;
  int ndx = 0;
  double sum = 0;

  // for every row of the matrix
  for( row = 0; row < NUM_FEATURES; row++ )
  {
    // multiply every matrix row element by every column element
    for( ndx = 0, sum = 0; ndx < NUM_FEATURES; ndx++ )
    {
      // perform the multiplication and add it to the sum
      sum += matrix[row][ndx] * column[ndx];
    }

    // store the result in the appropriate result column entry
    result_column[row] = sum;
  }

  // return the resulting column
  return result_column;
}

vector<double>& BayesClassifier::rowTimesMatrix( const vector<double>& row,
                                const vector<vector<double>>& matrix )
{
  // variables
  vector<double> result_row( NUM_FEATURES );
  int col = 0;
  int ndx = 0;
  double sum = 0;

  // for every column of the matrix
  for( col = 0; col < NUM_FEATURES; col++ )
  {
    // multiply each entry of the row by each entry of the
    // matrix column
    for( ndx = 0, sum = 0; sum < NUM_FEATURES; ndx++ )
    {
      // perform the multiplication and add the result to the sum
      sum += row[ndx] * matrix[ndx][col];
    }

    // store the resulting sum in the appropriate result row entry
    result_row[col] = sum;
  }

  // return the resulting row
  return result_row;
}

double BayesClassifier::rowTimesColumn( const vector<double>& row, const vector<double>& column )
{
  // variables
  double sum = 0;
  int ndx = 0;

  // visit each element of the vectors
  for( ndx = 0; ndx < NUM_FEATURES; ndx++ )
  {
    sum += row[ndx] * column[ndx];
  }

  // return the result
  return sum;
}


vector<double>& BayesClassifier::constantTimesVector( const double constant, const vector<double>& the_vector )
{
  // variables
  vector<double> result_vector( NUM_FEATURES );
  int ndx = 0;

  // multiply all values of the vector by the constant
  for( ndx = 0; ndx < NUM_FEATURES; ndx++ )
  {
    // do the multiplication
    result_vector[ndx] = constant * the_vector[ndx];
  }

  // return the resulting vector
  return result_vector;
}

double BayesClassifier::findChernoffBound( const double beta )
{
  return 0;
}

double BayesClassifier::findBhattacharyyaBound()
{
  return 0;
}



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   MUTATORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   ACCESSORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

