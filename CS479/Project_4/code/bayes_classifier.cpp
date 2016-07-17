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
BayesClassifier::BayesClassifier( const int num_features )
{
  // initialize all members
  mean_vector_.resize( num_features );
  mean_vector_ *= 0;
  covariance_matrix_.resize( num_features, num_features );
  covariance_matrix_ = Eigen::MatrixXd::Identity( num_features, num_features );
  inverse_covariance_matrix_.resize( num_features, num_features );
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
void BayesClassifier::train_classifier( const Eigen::MatrixXd& data )
{
    Eigen::VectorXd mean_vector;
    Eigen::MatrixXd covariance_matrix;
    Eigen::MatrixXd A_matrix;
    int j = 0;

    mean_vector.resize( data.rows(), 1 );
    mean_vector *= 0;

    for( j = 0; j < data.cols(); j++ )
    {
        mean_vector += data.col( j );
    }
    mean_vector /= data.cols();

    covariance_matrix.resize( mean_vector_.size(), mean_vector_.size() );
    A_matrix.resize( data.rows(), data.cols() );

    for( j = 0; j < data.cols(); j++ )
    {
        A_matrix.col( j ) = data.col( j ) - mean_vector;
    }

    covariance_matrix = A_matrix * A_matrix.transpose().eval();
    covariance_matrix *= 1.0 / (double) data.cols();

    // use the class' functionality to update all members/prevent
    // stale data
    set_mean_vector( mean_vector );
    set_covariance_matrix( covariance_matrix );

    assert( mean_vector_.size() == covariance_matrix_.rows() &&
            mean_vector_.size() == covariance_matrix_.cols() );

}


void BayesClassifier::set_mean_vector( const Eigen::VectorXd& new_mean_vector )
{
    mean_vector_.resize( new_mean_vector.size(), 1 );
    mean_vector_ = new_mean_vector;
}


void BayesClassifier::set_covariance_matrix( const Eigen::MatrixXd& new_covariance_matrix )
{
  assert( new_covariance_matrix.rows() == new_covariance_matrix.cols() );

  covariance_matrix_.resize( new_covariance_matrix.rows(), new_covariance_matrix.cols() );
  covariance_matrix_ = new_covariance_matrix;

  inverse_covariance_matrix_.resize( new_covariance_matrix.rows(), new_covariance_matrix.cols() );
  inverse_covariance_matrix_ = covariance_matrix_.inverse().eval();

  covariance_determinant_ = covariance_matrix_.determinant();
}


void BayesClassifier::set_prior_probability( const double new_probability )
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


Eigen::VectorXd BayesClassifier::mean_vector() const
{
  // return the prior mean feature vector of the class
  return mean_vector_;
}


Eigen::MatrixXd BayesClassifier::covariance_matrix() const
{
  // return the covariance matrix of the class
  return covariance_matrix_;
}


Eigen::MatrixXd BayesClassifier::inverse_covariance_matrix() const
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


void BayesClassifier::reportClassifierInfo()
{
  // variables
    // none

  // report the class name
  cout << "Classifier for class: " << class_name_ << endl;

  // report the trained mean
  printf( "The training data mean is:\r\n" );
  cout << mean_vector_ << endl;

  // report the trained covariance
  printf( "The training data covariance is:\r\n" );
  cout << covariance_matrix_ << endl;

  // report the prior probability
  printf( "The prior probability of the class is: %f\r\n\r\n",
          prior_probability_ );

  // no return - void
}


void BayesClassifier::performAnalysis(
    vector<BayesClassifier>& classifiers,
    vector<DataItem>& data,
    const string& output_file_name )
{
  // variables
  int i = 0;
  unsigned int num_correctly_classified = 0;
  double correct_classification_rate = 0;
  double battacharyya_bound;
  Chernoff chernoff_bound;
  fstream file;

  // classify all of the data
  for( i = 0; i < data.size(); i++ )
  {
    // assign the item to a class
    data[i].classified_as = assignToClass( data[i].feature_vector,
                                                   classifiers );

    // case: the data was classified correctly
    if( data[i].classified_as == data[i].actual_class )
    {
      // count the classification
      num_correctly_classified++;
      data[i].correctly_classified = CORRECT;
    }
    // case: the item was misclassified
    else
    {
      // indicate such
      data[i].correctly_classified = INCORRECT;
    }
  }

  // compute summary statistics
  correct_classification_rate = (double)  num_correctly_classified /
                                (double) data.size();
  battacharyya_bound = findBattacharyyaBound( classifiers );
  chernoff_bound = findChernoffBound( classifiers );

  // open the file to report the results
  file.clear();
  file.open( output_file_name.c_str(), fstream::out );

  // report the summary results
  file << "Analysis Results" << endl
       << "Number of items:               " << data.size() << endl
       << "Number correctly classified:   " << num_correctly_classified << endl
       << "Correct classification rate:   " << correct_classification_rate
                                            << endl
       << "Number incorrectly classified: " << ( data.size() -             
                                                 num_correctly_classified )
                                            << endl
       << "Classification error rate:     " << 1.0 - correct_classification_rate
                                            << endl
       << "Battacharyya bound:            " << battacharyya_bound << endl
       << "Chernoff bound:                " << chernoff_bound.bound << endl
       << "            b*:                " << chernoff_bound.beta_star << endl
       << endl;

  // summarize the classifiers used
  for( i = 0; i < classifiers.size(); i++ )
  {
    // report the data
    //classifiers[i].reportClassifierInfo();
  }

  // dump the data
  file << "Feature 1, Feature 2, Actual Class, Classified As, Correct?" << endl;
  for( i = 0; i< data.size(); i++ )
  {
    // write the delimited data to the file
    file << data[i].feature_vector(0) << ", "
         << data[i].feature_vector(1) << ", "
         << data[i].actual_class << ", "
         << data[i].classified_as << ", "
         << data[i].correctly_classified << endl;
  }

  // close the file
  file.close();

  // no return - void
}


string BayesClassifier::assignToClass(
    const Eigen::VectorXd& input_vector,
    vector<BayesClassifier>& classifiers )
{
  // variables
  string classification_result = "NO_RESULT";
  int i = 0;
  double discriminant = 0;
  double largest_discriminant = 0;
  double largest_discriminant_ndx = 0;

  // calculate the first discriminant
  largest_discriminant = classifiers[0].calculateDiscriminant( input_vector );

  // calculate each of the discriminants, keep track of the most likely class
  for( i = 1; i < classifiers.size(); ++i )
  {
    // calculate the next discriminant
    discriminant = classifiers[i].calculateDiscriminant( input_vector );

    // case: this discriminant is larger than the previously largest one
    if( largest_discriminant < discriminant )
    {
      // update the largest discriminant
      largest_discriminant = discriminant;

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
    const Eigen::VectorXd& input_vector )
{
  assert( input_vector.size() == mean_vector_.size() );

  // variables
  double discriminant_result = 0;
  double first_sum_term = 0;
  double second_sum_term = 0;
  double third_sum_term = 0;
  double fourth_sum_term = 0;
  double fifth_sum_term = 0;
  Eigen::VectorXd intermediate_row( input_vector.size() );
  Eigen::VectorXd intermediate_col( input_vector.size() );
  Eigen::MatrixXd intermediate_matrix( input_vector.size(), input_vector.size() );

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
  Eigen::VectorXd mean_difference;
  Eigen::MatrixXd covariance_sum;

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
  Eigen::VectorXd mean_difference;
    mean_difference.resize( classifiers[0].mean_vector().size(), 1 );
  Eigen::VectorXd intermediate_row;
  Eigen::MatrixXd scaled_covariance_sum;

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


