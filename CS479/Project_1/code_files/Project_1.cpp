/**
    @file Project_1.cpp

    @author Terence Henriod

    Project 1: Bayesion Minimum Error Classification

    @brief The driver program for use of a Bayesian Minimum Error Classifier.

    @version Original Code 1.00 (3/8/2014) - T. Henriod

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

  // end program
  return 0;
}

/*==============================================================================
=======     FUNCTION IMPLEMENTATIONS     =======================================
==============================================================================*/


