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
BayesClassifier

Description

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
  mean_vector_ << 1, 1;
  covariance_matrix_ << 1, 0,
                       0, 1;
  inverse_covariance_matrix_ = covariance_matrix_.inverse();
  covariance_determinant_ = covariance_matrix_.determinant();
  prior_probability_ = 0.5;
  class_name_ = "Give me a name!";

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
  // currently nothing to destruct

  // no return - destructor
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   MUTATORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


void BayesClassifier::clear()
{
  // no return - void
}


void BayesClassifier::setMean( const Eigen::Vector2d& new_mean_vector )
{
  // set the appropriate mean vector
  mean_vector_ = new_mean_vector;

  // no return - void
}


void BayesClassifier::setMean( const vector<DataItem>& data )
{
  // variables
  Eigen::Vector2d new_mean;
    new_mean << 0, 0;
  int i = 0;
  int num_data = 0;

  // sum the values of the features over all of the data
  for( i = 0; i < data.size(); i++ )
  {
    // case: the data is of the desired class
    if( data[i].actual_class == class_name_ )
    {
      // add the data to the sum
      new_mean( 0 ) += data[i].feature_vector( 0 );
      new_mean( 1 ) += data[i].feature_vector( 1 );
      num_data++;
    }
  }

  // scale the data
  new_mean( 0 ) /= num_data;
  new_mean( 1 ) /= num_data;

  // update the new mean member
  mean_vector_ = new_mean;

  // no return - void
}




void BayesClassifier::setCovariance(
    const Eigen::Matrix2d& new_covariance_matrix )
{
  // set the appropriate covariance matrix
  covariance_matrix_ = new_covariance_matrix;

  // update the other covariance related members
  inverse_covariance_matrix_ = covariance_matrix_.inverse();
  covariance_determinant_ = covariance_matrix_.determinant();

  // no return - void
}



void BayesClassifier::setCovariance( const vector<DataItem>& data,
                                     const Eigen::Vector2d& mean )
{
  // variables
  Eigen::Matrix2d new_covariance;
     new_covariance << 0, 0,
                       0, 0;
  double temp = 0;
  int i = 0;
  int num_data = 0;


  // sum the values of the features over all of the data
  for( num_data = 0, temp = 0, i = 0; i < data.size(); i++ )
  {
    // case: the data is of the desired class
    if( data[i].actual_class == class_name_ )
    {
      // add the data to the sum
      temp += ( data[i].feature_vector( 0 ) - mean( 0 ) ) *
              ( data[i].feature_vector( 0 ) - mean( 0 ) );
      num_data++;
    }
  }

  // scale the result
  new_covariance( 0, 0 ) /= (num_data - 1);


  // sum the values of the features over all of the data
  for( temp = 0, i = 0; i < data.size(); i++ )
  {
    // case: the data is of the desired class
    if( data[i].actual_class == class_name_ )
    {
      // add the data to the sum
      temp += ( data[i].feature_vector( 0 ) - mean( 0 ) ) *
              ( data[i].feature_vector( 1 ) - mean( 1 ) );
    }
  }

  // scale the result
  new_covariance( 0, 1 ) /= (num_data - 1);
  new_covariance( 1, 0 ) = new_covariance( 0, 1 );

  // sum the values of the features over all of the data
  for( temp = 0, i = 0; i < data.size(); i++ )
  {
    // case: the data is of the desired class
    if( data[i].actual_class == class_name_ )
    {
      // add the data to the sum
      temp += ( data[i].feature_vector( 1 ) - mean( 1 ) ) *
              ( data[i].feature_vector( 1 ) - mean( 1 ) );
    }
  }

  // scale the result
  new_covariance( 1, 1 ) /= (num_data - 1);

  // update the new mean member
  covariance_matrix_ = new_covariance;

  // update the other covariance related members
  inverse_covariance_matrix_ = covariance_matrix_.inverse();
  covariance_determinant_ = covariance_matrix_.determinant();

  // no return - void
}



void BayesClassifier::setPriorProbability( const double new_probability )
{
  // assert pre-conditions
  assert( ( new_probability >= 0.0 ) && ( new_probability <= 1.0 ) );

  // set the new prior probability
  prior_probability_ = new_probability;

  // no return - void
}

void BayesClassifier::set_class_name( const string& new_name )
{
  // set the class name member
  class_name_ = new_name;
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   ACCESSORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
double BayesClassifier::prior_probability() const
{
  // return the prior probability of the class
  return prior_probability_;
}


Eigen::Vector2d BayesClassifier::mean_vector() const
{
  // return the prior mean feature vector of the class
  return mean_vector_;
}


Eigen::Matrix2d BayesClassifier::covariance_matrix() const
{
  // return the covariance matrix of the class
  return covariance_matrix_;
}


Eigen::Matrix2d BayesClassifier::inverse_covariance_matrix() const
{
  // return the inverse of the covariance matrix of the class
  return inverse_covariance_matrix_;
}


double BayesClassifier::covariance_determinant() const
{
  // return the determinant of the covariance matrix of the class
  return covariance_determinant_;
}


string BayesClassifier::class_name() const
{
  // return the class name
  return class_name_;
}


void BayesClassifier::performAnalysis(
    const string input_file,
    const string output_file,
    const vector<BayesClassifier>& classifiers )
{

/*
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
    data[ndx].classified_as = assignToClass( data[ndx].feature_vector,
                                             classifiers );

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
*/
  // no return - void
}


string BayesClassifier::assignToClass(
    Eigen::Vector2d& input_vector,
    vector<BayesClassifier>& classifiers )
{
  // variables
  string classification_result = "NO_RESULT";
  int i = 0;
  double largest_discriminant = 0;
  double largest_discriminant_ndx = 0;

  // calculate each of the discriminants, keep track of the most likely class
  for( i = 0; i < classifiers.size(); ++i )
  {
    // case: this discriminant is larger than the previously largest one
    if( largest_discriminant <
        classifiers[i].calculateDiscriminant( input_vector ) )
    {
      // store the index of the class
      largest_discriminant_ndx = i;
    }
  }

  // whichever class produced the largest discriminant is the most likely
  classification_result = classifiers[largest_discriminant_ndx].class_name();

  // return the resulting assignment
  return classification_result;
}


double BayesClassifier::calculateDiscriminant(
    const Eigen::Vector2d& input_vector )
{
  // variables
  double discriminant_result = 0;
  double first_sum_term = 0;
  double second_sum_term = 0;
  double third_sum_term = 0;
  double fourth_sum_term = 0;
  double fifth_sum_term = 0;
  Eigen::Vector2d intermediate_row;
  Eigen::Vector2d intermediate_col;
  Eigen::Matrix2d intermediate_matrix;

  // compute the first summative term of the discriminant function
  intermediate_matrix = -0.5 * inverse_covariance_matrix_;
  intermediate_row = ( input_vector.transpose() * intermediate_matrix );
  first_sum_term = intermediate_row.dot( input_vector );

  // compute the second summative term of the discriminant function
  second_sum_term =
  ( inverse_covariance_matrix_ * mean_vector_ ).transpose().dot( input_vector );

  // compute the third summative term of the discriminant function
  intermediate_row = -0.5 * mean_vector_;
  intermediate_row = intermediate_row.transpose() * inverse_covariance_matrix_;
  third_sum_term = intermediate_row.dot( mean_vector_ );

  // compute the fourth summative term of the discriminant function
  fourth_sum_term = -0.5 * log( covariance_determinant_ );


/*  // case: we are assuming case 1 assumptions
  else
  {
    // compute the first term ( 1/s^2 * mean * x )
    intermediate_row = ( 1.0 / variance ) * mean;
    first_sum_term = intermediate_row.transpose().dot( input_vector );

    // compute the second term
    intermediate_row = ( -1.0 / ( 2 * variance ) ) * mean;
    second_sum_term = intermediate_row.transpose().dot( mean );
  }
 */

  // compute the last summative term of the discriminant function
  fifth_sum_term = log( prior_probability_ );


  // sum the terms to get the discriminant result
  discriminant_result = first_sum_term + second_sum_term + third_sum_term +
                        fourth_sum_term + fifth_sum_term;

  // return the discriminant result
  return discriminant_result;
}


Chernoff BayesClassifier::findChernoffBound(
    const vector<BayesClassifier>& classifiers )
{
  // variables
  Chernoff chernoff_bound;
    chernoff_bound.bound = 1.0;
    chernoff_bound.beta_star = 1.0;
  double beta = 0;
  double prior_product = 0.0;
  double kappa_of_beta = 0;
  double new_attempt;

  // test many possibilities to find the ideal beta*
  for( beta = 0.0; beta < 1.0; beta += EPSILON )
  {
    // compute the prior product
    prior_product = pow( classifiers[0].prior_probability(), beta ) *
                    pow( classifiers[1].prior_probability(), ( 1.0 - beta ) );

    // kappa( beta* )
    kappa_of_beta = kappaF( beta, classifiers );

    // attempt to find a lower bound
    new_attempt = prior_product * exp( -1.0 * kappa_of_beta );

    // case: the new bound is a tighter one
    if( new_attempt < chernoff_bound.bound )
    {
      // set the new bound and beta_star
      chernoff_bound.bound = new_attempt;
      chernoff_bound.beta_star = beta;
    }
  }

  // return the Chernoff bound
  return chernoff_bound;
}


double BayesClassifier::findBattacharyyaBound(
    const vector<BayesClassifier>& classifiers )
{
  // variables
  double battacharyya_bound = 1;
  double kappa_of_beta = 0;
  double root_prior_product = 0;
  double root_covariance_det_product = 0;
  Eigen::Vector2d mean_difference;
  Eigen::Matrix2d covariance_sum;

  // compute the square root term
  root_prior_product = sqrt( classifiers[0].prior_probability() *
                             classifiers[1].prior_probability() );

  // compute kappa( 0.5 )
  kappa_of_beta = kappaF( 0.5, classifiers );

  // compute sqrt( P( w1 ) * P( w2 ) ) * e^( -kappa( 0.5 ) )
  battacharyya_bound = root_prior_product * exp( -1.0 * kappa_of_beta );

  // return the Battacharrya bound
  return battacharyya_bound;
}


double BayesClassifier::kappaF( const double beta,
                                const vector<BayesClassifier>& classifiers )
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
  mean_difference = classifiers[0].mean_vector() - classifiers[1].mean_vector();

  // compute the scaled covariance sum beta^c * E1 + beta * E2
  // (to be used later)
  scaled_covariance_sum =
      ( beta_complement * classifiers[0].covariance_matrix() ) +
      ( beta * classifiers[1].covariance_matrix() );

  // compute the logarithm denominator (to be used later)
  log_denominator = pow( classifiers[0].covariance_determinant(),
                         beta_complement );
  log_denominator *= pow( classifiers[1].covariance_determinant(),
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


