/**
    @file bayes_classifier.cpp

    @author Terence Henriod

    Project 1: Bayesian Minimum Error Classification

    @brief Class implementations for the BayesClassifier defined in
           bayes_classifier.h.

    @version Original Code 1.00 (3/8/2014) - T. Henriod
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
#include <Eigen/Dense>  // -I /home/thenriod/Desktop/cpp_libs/Eigen_lib


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
  // variables
  int ndx = 0;
  Eigen::Vector2d temp_mean;
    temp_mean << 1, 1;
  Eigen::Matrix2d temp_matrix;
    temp_matrix << 1, 0,
                   0, 1;

  // initialize all members
  for( ndx = 0; ndx < NUM_FEATURES; ndx++ )
  {
    prior_probabilities.push_back( double( 1.0 / NUM_FEATURES ) );
  }
  for( ndx = 0; ndx < NUM_FEATURES; ndx++ )
  {
    mean_vectors.push_back( temp_mean );
  }
  for( ndx = 0; ndx < NUM_FEATURES; ndx++ )
  {
    covariance_matrices.push_back( temp_matrix );
  }
  assumption_case_ = CASE_THREE;

  // no return - constructor
}


BayesClassifier::BayesClassifier( const BayesClassifier& other )
{
  // no return - copy constructor
}


BayesClassifier& BayesClassifier::operator=( const BayesClassifier& other )
{
  // return *this
  return *this;
}


BayesClassifier::~BayesClassifier()
{
  // no return - destructor
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   MUTATORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


void BayesClassifier::clear()
{
  // no return - void
}


void BayesClassifier::setMean( const Eigen::Vector2d& new_mean_vector,
                               const int which_class )
{
  // variables
    // none

  // set the appropriate mean vector
  mean_vectors[which_class] = new_mean_vector;

  // no return - void
}


void BayesClassifier::setCovariance(
    const Eigen::Matrix2d& new_covariance_matrix,
    const int which_class )
{
  // variables
    // none

  // set the appropriate covariance matrix
  covariance_matrices[which_class] = new_covariance_matrix;

  // case: the new case was case 1
  if( assumption_case_ == CASE_ONE )
  {
    // update the variance
    variance = covariance_matrices[CLASS_ONE]( 0, 0 );
  }

  // no return - void
}


void BayesClassifier::setPriorProbabilities( const double class_one_prior )
{
  // assert pre-conditions
  assert( ( class_one_prior >= 0.0 ) && ( class_one_prior <= 1.0 ) );
  // assert( class_one_prior == ( 1 - class_one_prior ) );  // hopefully doubles don't screw this up
    // they do

  // variables
    // none

  // set the new prior probability
  prior_probabilities[0] = class_one_prior;
  prior_probabilities[1] = 1 - class_one_prior;

  // no return - void
}


void BayesClassifier::setAssumptionCase( const int new_case )
{
  // set the new case   TODO: automate this
  assumption_case_ = new_case;

  // case: the new case was case 1
  if( assumption_case_ == CASE_ONE )
  {
    // update the variance
    variance = covariance_matrices[CLASS_ONE]( 0, 0 );
  }
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   ACCESSORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void BayesClassifier::performAnalysis( const string input_file,
                                       const string output_file )
{
  // variables
  fstream file;
  TestData temp;
  char delimiter;
  int num_data = 0;
  int ndx = 0;
  int num_misclassified = 0;
  double test_error_rate = 0.5;
  double beta_start = 0.5;
  vector<TestData> data;
  Chernoff chernoff_bound;

  // read in all of the data
  file.clear();
  file.open( input_file.c_str(), fstream::in );
  while( file.good() )
  {
    // read in a line of data
    file >> temp.feature_vector(0) >> delimiter
         >> temp.feature_vector(1) >> delimiter
         >> temp.actual_class;

    // store the data
    data.push_back( temp );

    // count the data
    num_data++;
  }
  file.close();

  // classify all of the data
  for( ndx = 0; ndx < num_data; ndx++ )
  {
    // classify an object
    data[ndx].classified_as = assignToClass( data[ndx].feature_vector );

    // case: it was classified correctly
    if( data[ndx].classified_as == data[ndx].actual_class )
    {
      // mark and count this as a correct classification
      data[ndx].correctly_classified = CORRECT;
    }
    // case: it was not classified correctly
    else
    {
      // mark this as an incorrect classification
      data[ndx].correctly_classified = INCORRECT;
      num_misclassified++;
    }
  }

  // compute the error rate of the test
  test_error_rate = double( num_misclassified ) / double( num_data );

  // find chernoff bound
  chernoff_bound = findChernoffBound( beta_start, 0 );

  // output the results to file
  file.clear();
  file.open( output_file.c_str(), fstream::out );
  file << "Number of data: " << ( num_data - 1 ) << endl
       << "Number of incorrect classifications: "
           << num_misclassified << endl
       << "Test Sample Error Rate: " << test_error_rate << endl
       << "Battacharyya bound: " << findBattacharyyaBound() << endl
       << "Chernoff bound: " << chernoff_bound.bound << endl
       << "         beta*: " << chernoff_bound.beta << endl;
  for( ndx = 0; ndx < num_data; ndx++ )
  {
    // write the delimited data to the file
    file << data[ndx].feature_vector(0) << ", "
         << data[ndx].feature_vector(1) << ", "
         << data[ndx].actual_class << ", "
         << data[ndx].classified_as << ", "
         << data[ndx].correctly_classified << endl;
  }
  file.close();

  // no return - void
}


string BayesClassifier::assignToClass( Eigen::Vector2d& input_vector )
{
  // variables
  string classification_result;
  double discriminant_difference = 0;

  // calculate the difference of the discriminants
  discriminant_difference = calculateDiscriminant( input_vector, CLASS_ONE ) -
                            calculateDiscriminant( input_vector, CLASS_TWO );

  // case: the difference has a positive result
  if( discriminant_difference > 0 )
  {
    // the object is likely in class one
    classification_result = "ONE";
  }
  // case: the difference has a negative result
  else
  {
    // the object is likely in class two
    classification_result = "TWO";
  }

  // return the resulting assignment
  return classification_result;
}


double BayesClassifier::calculateDiscriminant(
    const Eigen::Vector2d& input_vector,
    const int which_class )
{
  // variables
  double discriminant_result = 0;
  double first_sum_term = 0;
  double second_sum_term = 0;
  double third_sum_term = 0;
  double fourth_sum_term = 0;
  double fifth_sum_term = 0;
  Eigen::Vector2d mean;
  Eigen::Matrix2d inverse_covariance_matrix;
  Eigen::Vector2d intermediate_row;
  Eigen::Vector2d intermediate_col;
  Eigen::Matrix2d intermediate_mat;

  // get the appropriate mean ready
  mean = mean_vectors[which_class];

  // prepare an inverse of the covariance matrix for the computations
  inverse_covariance_matrix = covariance_matrices[which_class].inverse();


  // case: the assumptions are not that of case 1
  if( assumption_case_ != CASE_ONE )
  {
    // compute the first summative term of the discriminant function
    intermediate_mat = -0.5 * inverse_covariance_matrix;
    intermediate_row = ( input_vector.transpose() * intermediate_mat );
    first_sum_term = intermediate_row.dot( input_vector );

    // compute the second summative term of the discriminant function
    second_sum_term = ( inverse_covariance_matrix * mean ).transpose().dot( input_vector );

    // compute the third summative term of the discriminant function
    intermediate_row = -0.5 * mean;
    intermediate_row = intermediate_row.transpose() * inverse_covariance_matrix;
    third_sum_term = intermediate_row.transpose().dot( mean );

    // compute the fourth summative term of the discriminant function
    fourth_sum_term = -0.5 *
                      log( covariance_matrices[which_class].determinant() );
  }
  // case: we are assuming case 1 assumptions
  else
  {
    // compute the first term ( 1/s^2 * mean * x )
    intermediate_row = ( 1.0 / variance ) * mean;
    first_sum_term = intermediate_row.transpose().dot( input_vector );

    // compute the second term
    intermediate_row = ( -1.0 / ( 2 * variance ) ) * mean;
    second_sum_term = intermediate_row.transpose().dot( mean );
  }

  // compute the last summative term of the discriminant function
  fifth_sum_term = log( prior_probabilities[which_class] );


  // sum the terms to get the discriminant result
  discriminant_result = first_sum_term + second_sum_term + third_sum_term +
                        fourth_sum_term + fifth_sum_term;

  // return the discriminant result
  return discriminant_result;
}


Chernoff BayesClassifier::findChernoffBound( double beta_star, int level )
{
  // variables
  Chernoff chernoff_bound;
    chernoff_bound.beta = beta_star;
  Chernoff left_attempt;
  Chernoff right_attempt;
  double beta_increment = 0.0249999;
  double prior_product = 0.0;
  double kappa_of_beta = 0;

  // compute the prior product
  prior_product = pow( prior_probabilities[0], beta_star ) *
                  pow( prior_probabilities[1], ( 1.0 - beta_star ) );

  // kappa( beta* )
  kappa_of_beta = kappaF( beta_star );

  // compute the Chernoff bound
  chernoff_bound.bound = prior_product * exp( -1.0 * kappa_of_beta );

  // case: we aren't 1000 levels deep
  if( level < 20 )
  {
    // find two different possible bounds
    left_attempt = findChernoffBound( beta_star - beta_increment, level + 1 );
    right_attempt = findChernoffBound( beta_star + beta_increment, level + 1 );

    // test to find the lowest bound
    if( left_attempt.bound < chernoff_bound.bound )
    {
      chernoff_bound = left_attempt;
    }
    if( right_attempt.bound < chernoff_bound.bound )
    {
      chernoff_bound = right_attempt;
    }
  }

  // return the Chernoff bound
  return chernoff_bound;
}


double BayesClassifier::findBattacharyyaBound()
{
  // variables
  double battacharyya_bound = 1;
  double kappa_of_beta = 0;
  double root_prior_product = 0;
  double root_covariance_det_product = 0;
  Eigen::Vector2d mean_difference;
  Eigen::Matrix2d covariance_sum;

  // compute the square root term
  root_prior_product = sqrt( prior_probabilities[0] * prior_probabilities[1] );

  // compute kappa( 0.5 )
  kappa_of_beta = kappaF( 0.5 );

  // compute sqrt( P( w1 ) * P( w2 ) ) * e^( -kappa( 0.5 ) )
  battacharyya_bound = root_prior_product * exp( -1.0 * kappa_of_beta );

  // return the Battacharrya bound
  return battacharyya_bound;
}


double BayesClassifier::kappaF( const double beta )
{
  // variables
  double kappa_of_beta = 0;
  double beta_complement = 1.0 - beta;
  double beta_product_over_two = 0;
  double root_prior_product = 0;
  double root_covariance_det_product = 0;
  double log_denominator = 0;
  Eigen::Vector2d mean_difference;
  Eigen::Vector2d intermediate_row;
  Eigen::Matrix2d scaled_covariance_sum;

  // compute (beta * beta^c) / 2
  beta_product_over_two = ( beta * beta_complement ) / 2;

  // compute the mean difference u2 - u1 (to be used later)
  mean_difference = mean_vectors[0] - mean_vectors[1];

  // compute the scaled covariance sum beta^c * E1 + beta * E2
  // (to be used later)
  scaled_covariance_sum = ( beta_complement * covariance_matrices[0] ) +
                          ( beta * covariance_matrices[1] );

  // compute the logarithm denominator (to be used later)
  log_denominator = pow( covariance_matrices[0].determinant(),
                         beta_complement );
  log_denominator *= pow( covariance_matrices[1].determinant(),
                          beta );

  // compute the first term in the sum
  intermediate_row = beta_product_over_two * mean_difference;
  intermediate_row = intermediate_row.transpose() *
                     scaled_covariance_sum.inverse();
  kappa_of_beta = intermediate_row.transpose() * mean_difference;

  // compute the second term in the sum
  kappa_of_beta += 0.5 * log( scaled_covariance_sum.determinant() /
                              log_denominator );  

  // return the result
  return kappa_of_beta;
}


