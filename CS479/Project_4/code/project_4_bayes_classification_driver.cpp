#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <Eigen/Dense>  // -I /home/thenriod/Desktop/cpp_libs/Eigen_lib
#include "bayes_classifier.h"
using namespace std;
using namespace Eigen;

const int NO_ERRORS = 0;
//const char CORRECT = 'X';
//const char INCORRECT = '-';

typedef struct
{
    string actual_class;
    string classified_as;
    bool correctly_classified;
} ClassificationItem;

typedef struct
{
    int num_correct;
    int num_incorrect;
    int false_males;
    int false_females;
    vector< ClassificationItem > tested_data;
} Results;


int loadData( MatrixXd& class_1_data, MatrixXd& class_2_data,
              const char* data_filename, const int num_features );

VectorXd computeMeanVector( const MatrixXd& data );

MatrixXd computeCovarianceMatrix( const MatrixXd& data, const VectorXd mean_vector );

MatrixXd mergeDataMatrices( const MatrixXd& class_1_data, const MatrixXd& class_2_data );

Results classifyTestData( const MatrixXd& class_1_test_data,
                          const MatrixXd& class_2_test_data,
                          vector< BayesClassifier >& classifiers );

void outputResults( const char* output_filename, const Results& results );


int main( int argc, char** argv )
{
    int program_success = NO_ERRORS;
    MatrixXd class_1_data;
    MatrixXd class_2_data;
    MatrixXd test_data;
    BayesClassifier male_image_class;
    BayesClassifier female_image_class;
    vector< BayesClassifier > classifiers( 2 );
    Results results;

    if( argc == 1 )
    {
        cout << "Usage: project4  num_features_to_keep training_data_filename" << endl
             << "testing_filenam output_filename" << endl;
    }

    puts( "Loading Training Data..." );
    loadData( class_1_data, class_2_data, argv[2], atoi( argv[1] ) );

    puts( "Training Bayes Classifiers..." );
    classifiers[0].set_class_name( "Male Image" );
    classifiers[0].train_classifier( class_1_data );
    classifiers[0].set_prior_probability( 0.5 );

    classifiers[1].set_class_name( "Female Image" );
    classifiers[1].train_classifier( class_2_data );
    classifiers[1].set_prior_probability( 0.5 );

//double y=0;
//for( int i = 0; i < class_1_data.cols(); i++ )
//{
//    cout << class_1_data( 0, i ) << '\t';
//    y += class_1_data( 0, i );
//}

//cout << endl << ( y / (double) class_1_data.cols() ) << endl;
//  classifiers[0].reportClassifierInfo();
//    classifiers[1].reportClassifierInfo();

    puts( "Loading Testing Data..." );
    loadData( class_1_data, class_2_data, argv[3], atoi( argv[1] ) ); // re-using objects

    puts( "Classifying Test Data..." );
    results = classifyTestData( class_1_data, class_2_data, classifiers );

    puts( "Outputting results..." );
    outputResults( argv[4], results );

    return program_success;
}


int loadData( MatrixXd& class_1_data, MatrixXd& class_2_data,
              const char* data_filename, const int num_features )
{
    ifstream fin;
    vector< VectorXd > feature_data;
    VectorXd temp_vector;
    vector< int > labels;
    int label;
    double value;
    int num_in_class_1 = 0;
    int num_in_class_2 = 0;
    char buffer[1000];
    int i = 0;
    int j = 0;
    int k = 0;

    temp_vector.resize( num_features, 1 );

    fin.clear();
    fin.open( data_filename );

    fin >> label;
    while( fin.good() )
    {
        labels.push_back( label );
        if( label == -1 )
        {
            num_in_class_1++;
        }
        else if( label == +1 ) // possibly unnecessary test, only expect -1/+1
        {
            num_in_class_2++;
        }

        for( j = 0; j < num_features; j++ )
        {
            fin.getline( buffer, 999, ':' ); // eat the index information
            fin >> value;
            temp_vector( j ) = value;
        }

        feature_data.push_back( temp_vector );
        fin.getline( buffer, 999, '\n' ); // eat any excess features' values
        i++;

        fin >> label;
    }

    fin.close();

    assert( ( num_in_class_1 + num_in_class_2 ) == labels.size() );

    class_1_data.resize( num_features, num_in_class_1 );
    class_2_data.resize( num_features, num_in_class_2 );

    for( i = 0, j = 0, k = 0; i < feature_data.size(); i++ )
    {
        if( labels[i] == -1 )
        {
            class_1_data.col( j ) = feature_data[i].transpose().eval();
            j++;
        }
        else if( labels[i] == +1 )
        {
            class_2_data.col( k ) = feature_data[i];
            k++;
        }
    }

    return 0; 
}

VectorXd computeMeanVector( const MatrixXd& data )
{
    VectorXd mean_vector;
    int j = 0;

    mean_vector.resize( data.rows(), 1 );
    mean_vector *= 0;

    for( j = 0; j < data.cols(); j++ )
    {
        mean_vector += data.col( j );
    }
    mean_vector /= data.cols();

    return mean_vector;
}

MatrixXd computeCovarianceMatrix( const MatrixXd& data, const VectorXd mean_vector )
{
    MatrixXd covariance_matrix;
    MatrixXd A_matrix;
    int j = 0;

    covariance_matrix.resize( mean_vector.size(), mean_vector.size() );
    A_matrix.resize( data.rows(), data.cols() );

    for( j = 0; j < data.cols(); j++ )
    {
        A_matrix.col( j ) = data.col( j ) - mean_vector;
    }

    covariance_matrix = A_matrix * A_matrix.transpose().eval();
    covariance_matrix *= data.cols();

    return covariance_matrix;
}


MatrixXd mergeDataMatrices( const MatrixXd& class_1_data, const MatrixXd& class_2_data )
{
    assert( class_1_data.rows() == class_2_data.rows() );

    MatrixXd merged_data;
    int j = 0;
    int k = 0;

    merged_data.resize( class_1_data.rows(), ( class_1_data.cols() + class_2_data.cols() ) );

    for( k = 0, j = 0; k < class_1_data.cols(); k++, j++ )
    {
        merged_data.col( j ) = class_1_data.col( k );
    }

    for( k = 0; k < class_2_data.cols(); k++, j++ )
    {
        merged_data.col( j ) = class_1_data.col( k );
    }

    return merged_data;
}


Results classifyTestData( const MatrixXd& class_1_test_data,
                          const MatrixXd& class_2_test_data,
                          vector< BayesClassifier >& classifiers )
{
    Results results;
        results.num_correct = 0;
        results.num_incorrect = 0;
        results.false_males = 0;
        results.false_females = 0;
        results.tested_data.resize( class_1_test_data.cols() + class_2_test_data.cols() );
    ClassificationItem temp;
    int i = 0;
    int j = 0;

    for( i = 0, j = 0; j < class_1_test_data.cols(); i++, j++ )
    {
        results.tested_data[i].actual_class = classifiers[0].class_name();
        results.tested_data[i].classified_as = BayesClassifier::assignToClass( class_1_test_data.col( j ), classifiers );
        results.tested_data[i].correctly_classified = ( results.tested_data[i].actual_class == results.tested_data[i].classified_as );

        if( results.tested_data[i].correctly_classified )
        {
            results.num_correct++;
        }
        else
        {
            results.num_incorrect++;

            if( results.tested_data[i].classified_as == classifiers[0].class_name() )
            {
                results.false_males++;
            }
            else if( results.tested_data[i].classified_as == classifiers[1].class_name() ) // likely could just be an else
            {
                results.false_females++;
            }
        }
    }

    for( j = 0; j < class_2_test_data.cols(); i++, j++ )
    {
        results.tested_data[i].actual_class = classifiers[1].class_name();
        results.tested_data[i].classified_as = BayesClassifier::assignToClass( class_2_test_data.col( j ), classifiers );
        results.tested_data[i].correctly_classified = ( results.tested_data[i].actual_class == results.tested_data[i].classified_as );

        if( results.tested_data[i].correctly_classified )
        {
            results.num_correct++;
        }
        else
        {
            results.num_incorrect++;

            if( results.tested_data[i].classified_as == classifiers[0].class_name() )
            {
                results.false_males++;
            }
            else if( results.tested_data[i].classified_as == classifiers[1].class_name() ) // likely could just be an else
            {
                results.false_females++;
            }
        }
    }

    return results;
}


void outputResults( const char* output_filename, const Results& results )
{
    ofstream fout;
    int i = 0;

    fout.clear();
    fout.open( output_filename );

    fout << "Accuracy:             " << (double) results.num_correct / (double) (results.num_correct + results.num_incorrect) << endl
         << "Number of Test Items: " << results.num_correct + results.num_incorrect << endl
         << "Number Correct:       " << results.num_correct << endl
         << "Number Incorrect:     " << results.num_incorrect << endl
         << "Number False Males:   " << results.false_males << endl
         << "Number False Females: " << results.false_females << endl
         << endl;

    fout << "Actual Class, Classified As, Correctly Classified (0 = No, 1 = Yes)" << endl;
    for( i = 0; i < results.tested_data.size(); i++ )
    {
        fout << results.tested_data[i].actual_class << ", " << results.tested_data[i].classified_as << ", " << results.tested_data[i].correctly_classified << endl;
    }

    fout.close();
}


