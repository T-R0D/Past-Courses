/**
    @file bayes_classifier.h

    @author Terence Henriod

    @brief Class declarations for the StrictGaussianClassifier which can be used to
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
#ifndef ___STRICT_GAUSSIAN_CLASSIFIER_H___
#define ___STRICT_GAUSSIAN_CLASSIFIER_H___


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

#include "bayes_classifier.h"
#include <Eigen/Dense>  // -I /home/thenriod/Desktop/cpp_libs/Eigen_lib

using namespace std;


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   GLOBAL CONSTANTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#define NUM_FEATURES 2
#define CORRECT 'o'
#define INCORRECT 'X'
#define EPSILON 0.00001
const double PI = 3.14159265359;
const double DIMENSIONALITY = 2;


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
                   CLASS DEFINITION(S)
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
@class StrictGaussianClassifierClassifier

dfsa;jlkdfs;ljkfdsa
@var lsakjdlfdsj
*/
class StrictGaussianClassifier
{
 public:
  /*---   Constructor(s) / Destructor   ---*/
  StrictGaussianClassifier();
  StrictGaussianClassifier( const StrictGaussianClassifier& other );
  StrictGaussianClassifier& operator=( const StrictGaussianClassifier& other );
  ~StrictGaussianClassifier();

  /*---   Mutators   ---*/
  void clear();
  void set_mean( const Eigen::Vector2d& new_mean_vector );


  void set_mean( const vector<DataItem>& data );



  void set_covariance( const Eigen::Matrix2d& new_covariance_matrix );


  void set_covariance( const vector<DataItem>& data,
                       const Eigen::Vector2d& mean );



  void set_prior_probability( const double new_probability );


  void set_class_name( const string& new_name );

  void set_decision_threshold( const double new_threshold );

  /*---   Accessors   ---*/
  double prior_probability() const;
  Eigen::Vector2d mean_vector() const;
  Eigen::Matrix2d covariance_matrix() const;
  Eigen::Matrix2d inverse_covariance_matrix() const;
  double covariance_determinant() const;
  double decision_threshold() const;
  string class_name() const;
  void reportClassifierInfo();

  double getGaussianProbability( Eigen::Vector2d& test_vector );

  bool objectIsInThisClass( Eigen::Vector2d& test_vector );

  /*---   Common Utilities   ---*/


 protected:

 private:
  /*---   Data Members   ---*/
  string class_name_;
  Eigen::Vector2d mean_vector_;
  Eigen::Matrix2d covariance_matrix_;
  Eigen::Matrix2d inverse_covariance_matrix_;
  double covariance_determinant_;
  double decision_threshold_;
};

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   TERMINATING PREPROCESSOR DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#endif		// #ifndef strict_gaussian_classifier.h

