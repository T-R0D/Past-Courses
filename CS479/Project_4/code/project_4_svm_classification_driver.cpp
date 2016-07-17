#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include "svm.h"
using namespace std;

const int NO_ERRORS = 0;
const int KILL_PROG = 1;
const int STD_STR_LEN = 100;
const char CORRECT = 'X';
const char INCORRECT = '-';

typedef struct
{
    int actual_class;
    int classified_as;
    bool correctly_classified;
} ClassificationItem;

typedef struct
{
    double accuracy;
    int num_correct;
    int num_incorrect;
    int false_positives;
    int false_negatives;
    vector< ClassificationItem > tested_data;
} Results;

svm_problem* loadData( const char* data_filename, const int num_features );

Results performClassification( svm_model* model, svm_problem* test_data );

void outputResults( const char* output_filepath, Results& results, svm_parameter* svm_parameters );

void free_svm_problem( svm_problem* problem );

int main( int argc, char** argv )
{
    int program_success = NO_ERRORS;
    int num_features = 0;
    int kernel_type = POLY;
    double c_param = 1;
    double gamma_param = 0.5;
    char training_filename[STD_STR_LEN];
    char testing_filename[STD_STR_LEN];
    char results_filepath[STD_STR_LEN];
    svm_problem* training_data = NULL;
    svm_problem* testing_data = NULL;
    svm_parameter* svm_parameters = new svm_parameter;
        /* unsure if necessary, set to default values from readme just in case*/
        svm_parameters->svm_type = C_SVC;
		svm_parameters->kernel_type = POLY;
		svm_parameters->degree = 3;	/* for poly */
		svm_parameters->gamma = 0.311; //( 1 / atoi( argv[1] ) );/* for poly/rbf/sigmoid */
		svm_parameters->coef0 = 0;	/* for poly/sigmoid */
		svm_parameters->cache_size = 100; /* in MB */
		svm_parameters->eps = 0.001;	/* stopping criteria */
		svm_parameters->C = 1;	/* for C_SVC, EPSILON_SVR, and NU_SVR */
		svm_parameters->nr_weight = 0;		/* for C_SVC */
		int *weight_label = NULL;	/* for C_SVC */
		double* weight = NULL;		/* for C_SVC */
		svm_parameters->nu = 0.5;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
		svm_parameters->p = 0.1;	/* for EPSILON_SVR */
		svm_parameters->shrinking = 1;	/* use the shrinking heuristics */
		svm_parameters->probability = 0; /* do probability estimates */
    svm_model* model = NULL;
    Results classification_results;

    if( argc == 1 || argc > 8 )
    {
        cout << "Usage: project4  num_features_to_keep" << endl
             << " kernel_type(see svm.h or lib-svm README)" << endl
             << "c_param" << endl
             << "gamma_param" << endl
             << "training_data_filename" << endl
             << "testing_filenam" << endl
             << "output_file_DIRECTORY" << endl;
        cout << "Note: using data scaled by 'svm-scale' is ideal." << endl;
    }
    else
    {
        num_features = atoi( argv[1] );
        kernel_type = atoi( argv[2] );
        c_param = atof( argv[3] );
        gamma_param = atof( argv[4] );
        strcpy( training_filename, argv[5] );
        strcpy( testing_filename, argv[6] );
        strcpy( results_filepath, argv[7] );

        puts( "Training SVM...\n" );
        training_data = loadData( training_filename, num_features );
        svm_parameters->kernel_type = kernel_type;
        svm_parameters->C = c_param;
        svm_parameters->gamma = gamma_param;
        if( svm_check_parameter( training_data, svm_parameters ) != NULL )
        {
            puts( "BAD PARAMETERS FOR LIBSVM TRAINING\n" );
            return KILL_PROG;
        }
        model = svm_train( training_data, svm_parameters );

        puts( "Testing SVM...\n" );
        testing_data = loadData( testing_filename, num_features );
        classification_results = performClassification( model, testing_data );

        puts( "Outputting results...\n" );
        outputResults( results_filepath, classification_results, svm_parameters );

        puts( "Ending program...\n" );
        free_svm_problem( training_data );
        free_svm_problem( testing_data );
        svm_destroy_param( svm_parameters );
        svm_free_and_destroy_model( &model );
    }

    return program_success;
}


svm_problem* loadData( const char* data_filename, const int num_features )
{
    svm_problem* problem_data = new svm_problem;
    ifstream fin;
    vector< vector< double > > feature_data;
    vector< double > temp;
    vector< int > labels;
    int label;
    char buffer[1000];
    int i = 0;
    int j = 0;

    temp.resize( num_features );

    fin.clear();
    fin.open( data_filename );

    fin >> label;
    while( fin.good() )
    {
        labels.push_back( label );

        for( j = 0; j < num_features; j++ )
        {
            fin.getline( buffer, 999, ':' ); // eat the index information
            fin >> temp[j];
        }
        feature_data.push_back( temp );
        fin.getline( buffer, 999, '\n' ); // eat any excess features' values
        i++;

        fin >> label;
    }

    fin.close();

/*
    // attempt to scale data
    double largest = -99999;
    for( i = 0; i < feature_data.size(); i++ )
    {
        for( j = 0; j < num_features; j++ )
        {
            if( largest < abs( feature_data[i][j] ) )
            {
                largest = abs( feature_data[i][j] );
            }
        }
    }
    for( i = 0; i < feature_data.size(); i++ )
    {
        for( j = 0; j < num_features; j++ )
        {
            feature_data[i][j] /= largest;
        }
    }
 */


    problem_data->l = labels.size();
    problem_data->y = new double [labels.size()];
    problem_data->x = new svm_node* [labels.size()];

    for( i = 0; i < labels.size(); i++ )
    {
        problem_data->y[i] = labels[i];

        problem_data->x[i] = new svm_node [num_features + 1];
        for( j = 0; j < num_features; j++ )
        {
            problem_data->x[i][j].index = (j + 1); // lib svm indexes features from 1
            problem_data->x[i][j].value = feature_data[i][j];
        }
        // libsvm requires the feature value list terminator (-1, ?)
        problem_data->x[i][j].index = -1;
        problem_data->x[i][j].value = '?';

    }

    return problem_data; 
}


Results performClassification( svm_model* model, svm_problem* test_data )
{
    Results test_results;
    ClassificationItem temp;
    int i = 0;
    int j = 0;

    test_results.tested_data.resize( test_data->l );
    test_results.num_correct = 0;
    test_results.num_incorrect = 0;
    test_results.false_positives = 0;
    test_results.false_negatives = 0;

    for( i = 0; i < test_data->l; i++ )
    {
        temp.actual_class = (int) test_data->y[i];
        temp.classified_as = (int) svm_predict( model, test_data->x[i] );
        temp.correctly_classified = ( temp.actual_class == temp.classified_as );

        if( temp.correctly_classified )
        {
            test_results.num_correct++;
        }
        else
        {
            test_results.num_incorrect++;

            if( temp.classified_as == 1 )
            {
                test_results.false_positives++;
            }
            else
            {
                test_results.false_negatives++;
            }
        }

        test_results.tested_data[i] = temp;
    }

    test_results.accuracy = (double) test_results.num_correct / (double) test_data->l;

    return test_results;
}

void outputResults( const char* output_filepath, Results& results, svm_parameter* svm_parameters )
{
    char actual_filename[STD_STR_LEN];
    ofstream fout;
    int i = 0;

    sprintf( actual_filename,
             "%s%d_accuracy_%d_kernel_%d_c_%d_gamma_results.txt",
             output_filepath,
             (int) (results.accuracy * 10000),
             svm_parameters->kernel_type,
             (int) svm_parameters->C,
             (int) (svm_parameters->gamma * 10) );

    fout.clear();
    fout.open( actual_filename );

    fout << "Accuracy:         " << results.accuracy << endl
         << "Kernel Type:      " << svm_parameters->kernel_type << endl
         << "C value:          " << svm_parameters->C << endl
         << "Gamma value:      " << svm_parameters->gamma << endl
         << "Number Tested:    " << results.tested_data.size() << endl
         << "Number Correct:   " << results.num_correct << endl
         << "Number Incorrect: " << results.num_incorrect << endl
         << "False Males:      " << results.false_positives << endl
         << "False Females:    " << results.false_negatives << endl
         << endl;

    fout << "Actual Class, Classified As, Correctly Classified" << endl;
    for( i = 0; i < results.tested_data.size(); i++ )
    {
        fout << results.tested_data[i].actual_class << ", "
             << results.tested_data[i].classified_as << ", "
             << results.tested_data[i].correctly_classified << endl;
    }

    fout.close();
}

void free_svm_problem( svm_problem* problem )
{
    int i = 0;

    for( i = 0; i < problem->l; i++ )
    {
        free( problem->x[i] );
    }
    free( problem->x );
    free( problem->y );
}


