#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace std;

const int STD_STR_LEN = 100;
const int NO_ERRORS = 0;

typedef struct
{
    int label;
    vector< double > feature_values;
} SvmData;


vector< SvmData > readAndMergeData( int num_features, char* labels_filename, char* features_filename );

int createMergedDataFile( const vector< SvmData >& data, char* output_filename );

int main( int argc, char** argv )
{
    int program_success = NO_ERRORS;
    int num_features;
    char label_file[STD_STR_LEN];
    char value_file[STD_STR_LEN];
    char output_file[STD_STR_LEN];
    vector< SvmData > data;

    if( argc <= 1 || argc > 5 )
    {
        cout << "Usage: merge-data-for-libsvm  number_of_features_to_keep" << endl
             << "class_labels_filename feature_data_filename  output_filename" << endl;
    }
    else
    {
        num_features = atoi( argv[1] );
        strcpy( label_file, argv[2] );
        strcpy( value_file, argv[3] );
        strcpy( output_file, argv[4] );

        printf( "\n"
                "Reading data from\n"
                "Label file:  %s\n"
                "Values file: %s\n",
                label_file,
                value_file );
        data = readAndMergeData( num_features, label_file, value_file );

        printf( "Creating data file with labels and data merged in: %s\n",
                output_file );
        createMergedDataFile( data, output_file );

        puts( "New data file created.\n" );
    }

    return program_success;
}


vector< SvmData > readAndMergeData( int num_features, char* labels_filename, char* features_filename )
{
    vector< SvmData > data;
    SvmData temp;
    ifstream labels;
    ifstream features;
    int temp_label;
    int i = 0;
    int j = 0;

    temp.feature_values.resize( num_features );

    // get labels as well as find the number of data in the set
    labels.clear();
    labels.open( labels_filename );

    labels >> temp_label;
    while( labels.good() )
    {
        temp.label = temp_label;
        data.push_back( temp );
        labels >> temp_label;
    }

    labels.close();

    // read in the values of each feature
    features.clear();
    features.open( features_filename );

    for( i = 0; i < num_features; i++ )
    {
        for( j = 0; j < data.size(); j++ )
        {
            features >> data[j].feature_values[i];
        }
    }

    features.close();

    return data;
}


int createMergedDataFile( const vector< SvmData >& data, char* output_filename )
{
    int creation_success = NO_ERRORS;
    ofstream fout;
    int i = 0;
    int j = 0;

    fout.clear();
    fout.open( output_filename );


    for( i = 0; i < data.size(); i++ )
    {
#if 1
        if( data[i].label == 1 )
        {
            fout << "-1";
        }
        else if( data[i].label == 2 )
        {
            fout << "+1";
        }
        fout << ' ';
#endif
#if 0
        fout << data[i].label << ' ';
#endif
        for( j = 0; j < data[i].feature_values.size(); j++ )
        {
            fout << (j + 1) << ':' << data[i].feature_values[j] << ' ';
        }
        fout << endl;
    }

    return creation_success;
}
