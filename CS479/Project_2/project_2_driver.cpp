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

#include "bayes_classifier.h"
#include "strict_gaussian_classifier.h"
#include "my_ppm.h"
#include <Eigen/Dense>  // -I /home/thenriod/Desktop/cpp_libs/Eigen_lib

using namespace std;

/*==============================================================================
=======     USER DEFINED TYPES     =============================================
==============================================================================*/
typedef struct
{
  string input_file_name;
  string output_file_base;
  string class_one_name;
  string class_two_name;
} BayesProblemData;


typedef struct
{
  string training_photo;
  string training_reference;
  string training_output;
  string photo_one;
  string reference_one;
  string output_one;
  string photo_two;
  string reference_two;
  string output_two;
} SkinProblemData;


/*==============================================================================
=======     CONSTANTS / MACROS     =============================================
==============================================================================*/
const string SKIN = "SKIN";
const string NOT_SKIN = "INANIMATE OBJECT";

/*==============================================================================
=======     GLOBAL VARIABLES     ===============================================
==============================================================================*/
  // none

/*==============================================================================
=======     FUNCTION PROTOTYPES     ============================================
==============================================================================*/

void solveBayesClassificationProblem( BayesProblemData& info );

int readData( vector<DataItem>& data, const string& input_file_name );

void solveImageProblem( SkinProblemData& info);

void trainStrictGaussianClassifiers( StrictGaussianClassifier& classifier_one,
                                     StrictGaussianClassifier& classifier_two,
                                     string& photo_file_name,
                                     string& reference_file_name );

void readPpmToDataVectors( vector<DataItem>& rb_picture_data,
                           vector<DataItem>& cb_cr_picture_data,
                          string& photo_file_name,
                          string& reference_file_name );

void classifyPhotoPixels( StrictGaussianClassifier& classifier,
                          vector<DataItem>& data,
                          string& output_file_name );

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
  BayesProblemData problem_info;
  SkinProblemData face_detection_info;

  string photo_file_name = "Training_1.ppm";
  string reference_file_name = "ref1.ppm";
  StrictGaussianClassifier skin_finder;



  // solve problem 1
  problem_info.input_file_name = "P1_data.txt";
  problem_info.output_file_base = "P1_solution";
  problem_info.class_one_name = "ONE";
  problem_info.class_two_name = "TWO";
  solveBayesClassificationProblem( problem_info );

  // solve problem 2
  problem_info.input_file_name = "P2_data.txt";
  problem_info.output_file_base = "P2_solution";
  solveBayesClassificationProblem( problem_info );




  // solve image problem
  face_detection_info.training_photo     = "Training_1.ppm";
  face_detection_info.training_reference = "ref1.ppm";
  face_detection_info.training_output    = "test_output.txt";
  face_detection_info.photo_one          = "Training_3.ppm";
  face_detection_info.reference_one      = "ref3.ppm";
  face_detection_info.output_one         = "output_1.txt";
  face_detection_info.photo_two          = "Training_6.ppm";
  face_detection_info.reference_two      = "ref6.ppm";
  face_detection_info.output_two         = "output_2.txt";

  skin_finder.set_class_name( SKIN );

  solveImageProblem( face_detection_info );


  // end program
  return 0;
}

/*==============================================================================
=======     FUNCTION IMPLEMENTATIONS     =======================================
==============================================================================*/
void solveBayesClassificationProblem( BayesProblemData& info )
{
  // variables
  vector<BayesClassifier> classifiers( 2 );
  vector<DataItem> problem_data;

  // read in the data
  readData( problem_data, info.input_file_name );

  // set the classifier class names
  classifiers[0].set_class_name( info.class_one_name );
  classifiers[1].set_class_name( info.class_two_name );

  // find the means of the training data
  classifiers[0].set_mean( problem_data );
  classifiers[1].set_mean( problem_data );

  // find the covariances of the training data
  classifiers[0].set_covariance( problem_data, classifiers[0].mean_vector() );
  classifiers[1].set_covariance( problem_data, classifiers[1].mean_vector() );

  // set the prior probabilities
  classifiers[0].set_prior_probability( 0.5 );
  classifiers[1].set_prior_probability( 0.5 );

  // perform the classifications with the equal prior probabilities
  BayesClassifier::performAnalysis( classifiers, problem_data,
                                    info.output_file_base + "_part1.txt" );

  // perform the classifications with the differing priors
  classifiers[0].set_prior_probability( 0.3 );
  classifiers[1].set_prior_probability( 0.7 );
  BayesClassifier::performAnalysis( classifiers, problem_data,
                                    info.output_file_base + "_part2.txt" );

  // no return - void
}


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


void solveImageProblem( SkinProblemData& info)
{
  // variables
  StrictGaussianClassifier rb_skin_finder;
  StrictGaussianClassifier cb_cr_skin_finder;
  string dummy = "CbCr_output.txt";

  vector<DataItem> rb_data;
  vector<DataItem> cb_cr_data;

  rb_skin_finder.set_class_name( SKIN );
  cb_cr_skin_finder.set_class_name( SKIN );

  // train the classifier
  trainStrictGaussianClassifiers( rb_skin_finder, cb_cr_skin_finder,
                                 info.training_photo,
                                 info.training_reference );


/*
readPpmToDataVectors( rb_data, cb_cr_data, info.training_photo, info.training_reference );


classifyPhotoPixels( rb_skin_finder, rb_data, info.training_output );

classifyPhotoPixels( cb_cr_skin_finder, cb_cr_data, dummy );
 */


  // perform classification on the first data set
  readPpmToDataVectors( rb_data, cb_cr_data, info.photo_one, info.reference_one );
  classifyPhotoPixels( rb_skin_finder, rb_data, info.output_one );
  dummy = "CbCr";
  dummy += info.output_one;
  classifyPhotoPixels( cb_cr_skin_finder, cb_cr_data, dummy );

  // clear the data vectors between runs
  rb_data.clear();
  cb_cr_data.clear();

  // perform classification on the second data set
  readPpmToDataVectors( rb_data, cb_cr_data, info.photo_two, info.reference_two );
  classifyPhotoPixels( rb_skin_finder, rb_data, info.output_two );
  dummy = "CbCr";
  dummy += info.output_two;
  classifyPhotoPixels( cb_cr_skin_finder, cb_cr_data, dummy );


  // no return - void
}


void trainStrictGaussianClassifiers( StrictGaussianClassifier& classifier_one,
                                     StrictGaussianClassifier& classifier_two,
                                     string& photo_file_name,
                                     string& reference_file_name )
{
  // variables
  vector<DataItem> rb_data;
  vector<DataItem> cb_cr_data;

  // read in the training data
  readPpmToDataVectors( rb_data, cb_cr_data, photo_file_name,
                        reference_file_name );

  // find the training mean
  classifier_one.set_mean( rb_data );
  classifier_two.set_mean( cb_cr_data );

  // find the covariance
  classifier_one.set_covariance( rb_data, classifier_one.mean_vector() );
  classifier_two.set_covariance( cb_cr_data, classifier_two.mean_vector() );

cout << "RB Classifier" << endl;
classifier_one.reportClassifierInfo();

cout << "CbCr Classifier" << endl;
classifier_two.reportClassifierInfo();

  // return the trained classifier by reference
}


void readPpmToDataVectors( vector<DataItem>& rb_picture_data,
                            vector<DataItem>& cb_cr_picture_data,
                            string& photo_file_name,
                            string& reference_file_name )
{
  // variables
  PpmImageData photo_image;
  PpmImageData reference_image;
  DataItem temp;
  int i = 0;
  int j = 0;
  int k = 0;
  double color_sum = 0;

  // read in the image file
  readPpmFile( &photo_image, photo_file_name.c_str() );

  // read in the reference file
  readPpmFile( &reference_image, reference_file_name.c_str() );

  // visit each row of pixels in the images
  for( i = 0, k = 0; i < photo_image.height; i++ )
  {
    // visit each pixel of the rows
    for( j = 0; j < photo_image.width; j++, k++ )
    {
      // convert the pixel color vector to a two dimensional one
      color_sum = photo_image.data[i][j].red + photo_image.data[i][j].green +
                  photo_image.data[i][j].blue;
      if( color_sum > 0 )
      {
        temp.feature_vector( 0 ) = photo_image.data[i][j].red / color_sum;
        temp.feature_vector( 1 ) = photo_image.data[i][j].green / color_sum;
      }
      else  // to not divide by 0
      {
        temp.feature_vector( 0 ) = 0;
        temp.feature_vector( 1 ) = 0;
      }

      // case: the reference image indicates the pixel in the photo is skin
      if( (reference_image.data[i][j].red + reference_image.data[i][j].green +
           reference_image.data[i][j].blue) > 0 )
      {
        // tag the data
        temp.actual_class = SKIN;
      }
      // case: the photo pixel is non-skin
      else
      {
        // tag the data
        temp.actual_class = NOT_SKIN;
      }

      // add the new item to the vector
      rb_picture_data.push_back( temp );

      // compute the Cb and Cr values for the alternative
      temp.feature_vector( 0 ) = ( -0.169 * photo_image.data[i][j].red ) +
                                 ( -0.332 * photo_image.data[i][j].green ) +
                                 ( 0.5 * photo_image.data[i][j].blue );
      temp.feature_vector( 1 ) = ( 0.5 * photo_image.data[i][j].red ) +
                                 ( -0.419 * photo_image.data[i][j].green ) +
                                 ( -0.081 * photo_image.data[i][j].blue );

      // add the new item to the new vector
      cb_cr_picture_data.push_back( temp );
    }
  }

  // deconstruct the image structs
  deconstructPpmImage( &photo_image );
  deconstructPpmImage( &reference_image );

  // return the data vector by reference
}


void classifyPhotoPixels( StrictGaussianClassifier& classifier,
                          vector<DataItem>& data,
                          string& output_file_name )
{
  // variables
  vector<double> skin_likelihood;
  int i = 0;
  int num_skin = 0;
  int num_other = 0;
  int num_correct_skin = 0;
  int num_correct_other = 0;
  int false_acceptance = 0;
  int false_rejection = 0;
  double threshold = 0;
  double overall_correct_rate = 0;
  double skin_correct_rate = 0;
  double other_correct_rate = 0;
  double false_acceptance_rate = 0;
  double false_rejection_rate = 0;
  fstream file;

  // prepare the file for the output summary
  file.clear();
  file.open( output_file_name.c_str(), fstream::out );
  file << "Threshold Used, "
       << "Number of items, "
       << "Correct Classifications, "
       << "Correct Classification Rate, "
       << "Number of Skin Items, "
       << "Correct Skin Classifications, "
       << "Correct Skin Classification Rate, "
       << "Number of Other Items, "
       << "Correct Other Classifications, "
       << "Correct Other Classification Rate, "
       << "False Acceptance Rate, "
       << "False Rejection Rate"<< endl;

  // find the likelihood of each test vector being a skin one
  for( i = 0; i < data.size(); i++ )
  {
     // compute the likelihood that the pixel is skin
     skin_likelihood.push_back(
         classifier.getGaussianProbability( data[i].feature_vector ) );
  }


  // perform the classifications at various acceptance thresholds
  for( threshold = 0.01; threshold < 1.0; threshold += 0.01 )
  {
    // set the decision threshold
    classifier.set_decision_threshold( threshold );

    // reset the counters
    num_skin = 0;
    num_other = 0;
    num_correct_skin = 0;
    num_correct_other = 0;
    false_acceptance = 0;
    false_rejection = 0;

    // for every skin-pixel likelihood
    for( i = 0; i < data.size(); i++ )
    {
       // case: the pixel is likely skin
       if( skin_likelihood[i] > threshold )
       {
          // mark the pixel as skin
          data[i].classified_as = SKIN;

          // case: the pixel was not correctly classified
          if( data[i].actual_class != SKIN  )
          {
            // count it
            false_acceptance++;
          }
          else
          {
             num_correct_skin++;
          }
       }
       // case: the skin pixel is not likely skin
       else
       {
          // mark the pixel as skin
          data[i].classified_as = NOT_SKIN;

          // case: the pixel was actually skin
          if( data[i].actual_class == SKIN )
          {
            // count it
            false_rejection++;
          }
          else
          {
            num_correct_other++;
          }
       }

       // case: the pixel was in fact skin
       if( data[i].actual_class == SKIN )
       {
         // count it
         num_skin++;
       }
       else
       {
         num_other++;
       }
    }

    // compute the summary statistics
    overall_correct_rate = (double)(num_correct_skin + num_correct_other) /
                           (double) data.size();
    skin_correct_rate = (double)((double)num_correct_skin / (double)num_skin);
    other_correct_rate = (double)((double)num_correct_other /
                         (double)num_other);
    false_acceptance_rate = (double) false_acceptance / (double) (data.size() - num_skin);  //false positive rate = false positive / negative
    false_rejection_rate = (double) false_rejection / (double) num_skin;    // false negative rate = false negative / positive

    // output the result data to the output file
    file << classifier.decision_threshold() << ", "
         << data.size() << ", "
         << (num_correct_skin + num_correct_other) << ", "
         << overall_correct_rate << ", "
         << num_skin << ", "
         << num_correct_skin << ", "
         << skin_correct_rate << ", "
         << num_other << ", "
         << num_correct_other << ", "
         << other_correct_rate << ", "
         << false_acceptance_rate << ", "
         << false_rejection_rate
         << endl;
  }

  // close the file
  file.close();

  // no return - void
}


