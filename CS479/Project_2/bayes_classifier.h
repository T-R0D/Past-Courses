/**
    @file bayes_classifier.h

    @author Terence Henriod

    @brief Class declarations for the BayesClassifier which can be used to
           use Bayes Minimum Average Error classifacation for two categories
           as well as specify Chernoff and Battacharyya upper error bounds.

    @version Original Code 1.00 (3/26/2014) - T. Henriod

    UNOFFICIALLY:
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
#define EPSILON 0.00001


typedef struct
{
  Eigen::Vector2d feature_vector;
  string actual_class;
  string classified_as;
  char   correctly_classified;
} DataItem;

typedef struct
{
  double bound;
  double beta_star;
} Chernoff;


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
  void set_mean( const Eigen::Vector2d& new_mean_vector );


  void set_mean( const vector<DataItem>& data );



  void set_covariance( const Eigen::Matrix2d& new_covariance_matrix );


  void set_covariance( const vector<DataItem>& data,
                      const Eigen::Vector2d& mean );



  void set_prior_probability( const double new_probability );


  void set_class_name( const string& new_name );

  /*---   Accessors   ---*/
  double prior_probability() const;
  Eigen::Vector2d mean_vector() const;
  Eigen::Matrix2d covariance_matrix() const;
  Eigen::Matrix2d inverse_covariance_matrix() const;
  double covariance_determinant() const;
  string class_name() const;
  void reportClassifierInfo();

  /*---   Common Utilities   ---*/
  static void performAnalysis( vector<BayesClassifier>& classifiers,
                               vector<DataItem>& data,
                               const string& output_file_name );

  static string assignToClass( const Eigen::Vector2d& input_vector,
                               vector<BayesClassifier>& classifiers );

  double calculateDiscriminant( const Eigen::Vector2d& input_vector );

  static Chernoff findChernoffBound(
      const vector<BayesClassifier>& classifiers );

  static double findBattacharyyaBound(
      const vector<BayesClassifier>& classifiers );

  static double kappaF( const double beta,
                        const vector<BayesClassifier>& classifiers );


 protected:

 private:
  /*---   Data Members   ---*/
  double prior_probability_;
  Eigen::Vector2d mean_vector_;
  Eigen::Matrix2d covariance_matrix_;
  Eigen::Matrix2d inverse_covariance_matrix_;
  double covariance_determinant_;
  string class_name_;
};

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   TERMINATING PREPROCESSOR DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#endif		// #ifndef bayes_classifier.h
