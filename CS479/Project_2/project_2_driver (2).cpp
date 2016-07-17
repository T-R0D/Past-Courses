/**
    @file Project_2.cpp

    @author Terence Henriod

    Project 2: Bayesion Minimum Error Classification

    @brief The driver program for use of a Bayesian Minimum Error Classifier to
           both classify randomly generated data and detect face (or at least
           skin-colored) regions in images.

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



Compilation notes:
g++ -I /home/thenriod/Desktop/cpp_libs/Eigen_lib/ Project_1.cpp

*/

/*==============================================================================
=======     HEADER FILES     ===================================================
==============================================================================*/
#include <cmath>
#include <iostream>

#include "bayes_classifier.cpp"
#include <Eigen/Dense>  // -I /home/thenriod/Desktop/cpp_libs/Eigen_lib

using namespace std;

/*==============================================================================
=======     USER DEFINED TYPES     =============================================
==============================================================================*/



/*==============================================================================
=======     CONSTANTS / MACROS     =============================================
==============================================================================*/


/*==============================================================================
=======     GLOBAL VARIABLES     ===============================================
==============================================================================*/
  // none

/*==============================================================================
=======     FUNCTION PROTOTYPES     ============================================
==============================================================================*/
int readData( vector<DataItem>& data, const string& input_file_name );

/*==============================================================================
=======     MAIN FUNCTION     ==================================================
==============================================================================*/

/**
main

The main driver

@param

@return

@pre
-#

@post
-#

@code
@endcode
*/

int main( int argc, char** argv )
{
  // variables
  vector<BayesClassifier> classifiers( 2 );
  vector<DataItem> problem_data;

  // solve problem 1
    // read in the data
    readData( problem_data, "P1_data.txt" );

    // set the classifier class names
    classifiers[0].set_class_name( "ONE" );
    classifiers[1].set_class_name( "TWO" );

    cout << classifiers[0].class_name() << endl;
    cout << classifiers[1].class_name() << endl;

    // set the prior probabilities
    classifiers[0].setPriorProbability( 0.5 );
    classifiers[1].setPriorProbability( 0.5 );

    cout << classifiers[0].prior_probability() << endl;
    cout << classifiers[1].prior_probability() << endl;

    // find the means and covariances
    classifiers[0].setMean( problem_data );
    classifiers[1].setMean( problem_data );

    cout << (classifiers[0].mean_vector())( 0 ) << endl;
    cout << (classifiers[0].mean_vector())( 1 ) << endl;

    cout << (classifiers[1].mean_vector())( 0 ) << endl;
    cout << (classifiers[1].mean_vector())( 1 ) << endl;


    // perform the classifications

    // report results

  // solve problem 2

  // solve image problem
















/*
  BayesClassifier problem_solver;
  double part_A_prior_probability = 0.5;
  double part_B_prior_probability = 0.3;
  Eigen::Vector2d part_one_mean_one;
    part_one_mean_one << 1.502, 1.484;
  Eigen::Vector2d part_one_mean_two;
    part_one_mean_two << 2.499, 2.497;
  Eigen::Vector2d part_two_mean_one;
    part_two_mean_one << 1, 2;
  Eigen::Vector2d part_two_mean_two;
    part_two_mean_two << 1, 4;
  Eigen::Matrix2d part_one_covariance_one;
    part_one_covariance_one << 1.099, 0,
                               0, 1.099;
  Eigen::Matrix2d part_one_covariance_two;
    part_one_covariance_two << 1.813, 0,
                               0, 1.813;
  Eigen::Matrix2d part_two_covariance_one;
    part_two_covariance_one << 1, 0,
                               0, 1;
  Eigen::Matrix2d part_two_covariance_two;
    part_two_covariance_two << 3, 0,
                               0, 2;

  // setup for problem 1
  problem_solver.setMean( part_one_mean_one, CLASS_ONE );
  problem_solver.setMean( part_one_mean_two, CLASS_TWO );
  problem_solver.setCovariance( part_one_covariance_one, CLASS_ONE );
  problem_solver.setCovariance( part_one_covariance_two, CLASS_TWO );
  problem_solver.setPriorProbabilities( part_A_prior_probability );
  problem_solver.setAssumptionCase( CASE_ONE );

  // solve 1.a
//  problem_solver.performAnalysis( "P1_data.txt", "test_1A_output.txt" );
problem_solver.performAnalysis( "Bebis11.txt", "Bebis1A.txt" );

  // solve 1.b
  problem_solver.setPriorProbabilities( part_B_prior_probability );
//  problem_solver.performAnalysis( "P1_data.txt", "test_1B_output.txt" );
problem_solver.performAnalysis( "Bebis11.txt", "Bebis1B.txt" );

  // setup for problem 2
  problem_solver.setMean( part_two_mean_one, CLASS_ONE );
  problem_solver.setMean( part_two_mean_two, CLASS_TWO );
  problem_solver.setCovariance( part_two_covariance_one, CLASS_ONE );
  problem_solver.setCovariance( part_two_covariance_two, CLASS_TWO );
  problem_solver.setPriorProbabilities( part_A_prior_probability );
  problem_solver.setAssumptionCase( CASE_THREE );

  // solve 2.a
//  problem_solver.performAnalysis( "P2_data.txt", "test_2A_output.txt" );
problem_solver.performAnalysis( "Bebis21.txt", "Bebis2A.txt" );

  // solve 2.b
  problem_solver.setPriorProbabilities( part_B_prior_probability );
//  problem_solver.performAnalysis( "P2_data.txt", "test_2B_output.txt" );
problem_solver.performAnalysis( "Bebis21.txt", "Bebis2B.txt" );
*/

  // end program
  return 0;
}

/*==============================================================================
=======     FUNCTION IMPLEMENTATIONS     =======================================
==============================================================================*/

int readData( vector<DataItem>& data, const string& input_file_name )
{
  // variables
  fstream file;
  DataItem temp;
  char delimiter;

  // clear file stream object and open the file
  file.clear();
  file.open( input_file_name.c_str(), fstream::in );

  // prime the reading loop
  file >> temp.feature_vector(0) >> delimiter
       >> temp.feature_vector(1) >> delimiter
       >> temp.actual_class;

  // continue to read from the file while possible
  while( file.good() )
  {
    // store the recently read data
    data.push_back( temp );

    // attempt to read more data
    file >> temp.feature_vector(0) >> delimiter
         >> temp.feature_vector(1) >> delimiter
         >> temp.actual_class;
  }

  // return the data vector by reference
}
