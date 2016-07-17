/**
    @file bayes_classifier.cpp

    @author Terence Henriod

    Project 1: Bayesian Minimum Error Classification

    @brief Class implementations for the StrictGaussianClassifier defined in
           bayes_classifier.h.

    @version Original Code 1.00 (3/8/2014) - T. Henriod
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// Class Declaration
#include "strict_gaussian_classifier.h"

// Other Dependencies
#include <cassert>
#include <iostream>
#include <fstream>

#include "bayes_classifier.h"
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
StrictGaussianClassifier

Description

@pre
-# The GameState object is given an appropriate identifier.

@post
-# A new, empty GameState will be initialized.

@code
@endcode
*/
StrictGaussianClassifier::StrictGaussianClassifier()
{
  // variables
  int ndx = 0;
  Eigen::Matrix2d temp_matrix;
    temp_matrix << 1, 0,
                   0, 1;

  // initialize all members
  class_name_ = "Give me a name!";
  mean_vector_ << 1, 1;
  set_covariance( temp_matrix );
  decision_threshold_ = 0.5;

  // no return - constructor
}


StrictGaussianClassifier::StrictGaussianClassifier(
    const StrictGaussianClassifier& other )
{
  // no return - copy constructor
}


StrictGaussianClassifier& StrictGaussianClassifier::operator=(
    const StrictGaussianClassifier& other )
{
  // return *this
  return *this;
}


StrictGaussianClassifier::~StrictGaussianClassifier()
{
  // currently nothing to destruct

  // no return - destructor
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   MUTATORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


void StrictGaussianClassifier::clear()
{
  // no return - void
}


void StrictGaussianClassifier::set_mean( const Eigen::Vector2d& new_mean_vector )
{
  // set the appropriate mean vector
  mean_vector_ = new_mean_vector;

  // no return - void
}


void StrictGaussianClassifier::set_mean( const vector<DataItem>& data )
{
  // variables
  int i = 0;
  int num_data = 0;

  // reset the mean vector
  mean_vector_( 0 ) = 0;
  mean_vector_( 1 ) = 0;

  // sum the values of the features over all of the data
  for( i = 0; i < data.size(); i++ )
  {
    // case: the data is of the desired class
    if( data[i].actual_class == class_name_ )
    {
      // add the data to the sum
      mean_vector_( 0 ) += data[i].feature_vector( 0 );
      mean_vector_( 1 ) += data[i].feature_vector( 1 );
      num_data++;
    }
  }

  // scale the data
  mean_vector_( 0 ) /= num_data;
  mean_vector_( 1 ) /= num_data;

  // no return - void
}


void StrictGaussianClassifier::set_covariance(
    const Eigen::Matrix2d& new_covariance_matrix )
{
  // set the appropriate covariance matrix
  covariance_matrix_ = new_covariance_matrix;

  // update the other covariance related members
  inverse_covariance_matrix_ = covariance_matrix_.inverse();
  covariance_determinant_ = covariance_matrix_.determinant();

  // no return - void
}


void StrictGaussianClassifier::set_covariance( const vector<DataItem>& data,
                                      const Eigen::Vector2d& mean )
{
  // variables
  int i = 0;
  int num_data = 0;

  // reset the covariance matrix
  covariance_matrix_ << 0, 0,
                        0, 0;


  // sum the values of the features over all of the data
  for( num_data = 0, i = 0; i < data.size(); i++ )
  {
    // case: the data is of the desired class
    if( data[i].actual_class == class_name_ )
    {
      // add the data to the sums
      covariance_matrix_( 0, 0 ) +=
          ( data[i].feature_vector( 0 ) - mean( 0 ) ) *
          ( data[i].feature_vector( 0 ) - mean( 0 ) );
      covariance_matrix_( 1, 0 ) +=
          ( data[i].feature_vector( 1 ) - mean( 1 ) ) *
          ( data[i].feature_vector( 0 ) - mean( 0 ) );
      covariance_matrix_( 1, 1 ) +=
          ( data[i].feature_vector( 1 ) - mean( 1 ) ) *
          ( data[i].feature_vector( 1 ) - mean( 1 ) );
      num_data++;
    }
  }

  // set the covariance above the diagonal
  covariance_matrix_( 0, 1 ) = covariance_matrix_( 1, 0 );

  // scale the result
  covariance_matrix_ = ( 1.0 / ( (double) num_data - 1.0) ) *
                       covariance_matrix_;

  // update the other covariance related members
  inverse_covariance_matrix_ = covariance_matrix_.inverse();
  covariance_determinant_ = covariance_matrix_.determinant();

  // no return - void
}


void StrictGaussianClassifier::set_class_name( const string& new_name )
{
  // set the class name member
  class_name_ = new_name;
}


void StrictGaussianClassifier::set_decision_threshold(
    const double new_threshold )
{
  // set the class decision threshold member
  decision_threshold_ = new_threshold;
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   ACCESSORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

Eigen::Vector2d StrictGaussianClassifier::mean_vector() const
{
  // return the prior mean feature vector of the class
  return mean_vector_;
}


Eigen::Matrix2d StrictGaussianClassifier::covariance_matrix() const
{
  // return the covariance matrix of the class
  return covariance_matrix_;
}


Eigen::Matrix2d StrictGaussianClassifier::inverse_covariance_matrix() const
{
  // return the inverse of the covariance matrix of the class
  return inverse_covariance_matrix_;
}


double StrictGaussianClassifier::covariance_determinant() const
{
  // return the determinant of the covariance matrix of the class
  return covariance_determinant_;
}


string StrictGaussianClassifier::class_name() const
{
  // return the class name
  return class_name_;
}

double StrictGaussianClassifier::decision_threshold() const
{
  // return the decision threshold value
  return decision_threshold_;
}

void StrictGaussianClassifier::reportClassifierInfo()
{
  // variables
    // none

  // report the class name
  cout << "Classifier for class " << class_name_ << endl;

  // report the trained mean
  printf( "The training data mean is:\r\n" );
  cout << mean_vector_ << endl;

  // report the trained covariance
  printf( "The training data covariance is:\r\n" );
  cout << covariance_matrix_ << endl;

  // no return - void
}


bool StrictGaussianClassifier::objectIsInThisClass( Eigen::Vector2d& test_vector )
{
  // return the decision that the object is in this class
  return ( getGaussianProbability( test_vector ) > decision_threshold_ );
}


double StrictGaussianClassifier::getGaussianProbability(
    Eigen::Vector2d& test_vector )
{
  // variables
  double gaussian_probability_density = 0;
  double fractional_part = 1;
  double exponent_part = 0;
  Eigen::Vector2d test_mean_difference;
  Eigen::Vector2d intermediate_vector;

/*
    DON'T KNOW WHY, BUT THE SCALE FACTOR IS RUINING THINGS - IS DATA ALREADY
    NORMALIZED/SCALED SOMEHOW? 

  // compute the normalizing/scale factor
  fractional_part = sqrt( 2 * PI );
  fractional_part = pow( fractional_part, DIMENSIONALITY );
  fractional_part *= sqrt( covariance_determinant_ );
  fractional_part = pow( fractional_part, -1 );
*/

  // compute the expontent part
  test_mean_difference = test_vector - mean_vector_;

  intermediate_vector = test_mean_difference.transpose() *
                        inverse_covariance_matrix_;
  exponent_part = -0.5 * ( intermediate_vector.dot( test_mean_difference ) );

  // compute the whole thing
  gaussian_probability_density = fractional_part * exp( exponent_part );

  // return the result
  return gaussian_probability_density;
}


