/**
    @file .cpp

    @author Terence Henriod

    Project Name

    @brief This program...

    @version Original Code 1.00 (10/29/2013) - T. Henriod
*/

/*
bool extensionIsGood( const string& file_name, const string& extension )
{
    return ( file_name.find( extension, file_name.length() - 5 ) !=
             string::npos );
}
 */

/*==============================================================================
=======     HEADER FILES     ===================================================
==============================================================================*/
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>

#include <Eigen/Dense>  // -I /home/thenriod/Desktop/cpp_libs/Eigen_lib
#include "my_stopwatch.h"

using namespace std;
using namespace Eigen;



/*==============================================================================
=======     GLOBAL CONSTANTS     ===============================================
==============================================================================*/
#define DEBUG 1

const int STD_STR_LEN = 100;

const char MATCH = 'X';
const char NO_MATCH = '-';

const string TRAINING_RESULTS_DIRECTORY = "/training_results/";

enum Status
{
    NO_ERRORS = 0,
    ERROR
};

enum Mode
{
    TRAINING_MODE = 0,
    TESTING_MODE = 1,
    INVALID = 666
};


/*==============================================================================

=======     USER DEFINED TYPES     =============================================
==============================================================================*/

/**
@struct

Description

@var
*/
typedef struct
{
    char** training_image_names;
    int num_training_images;
    char** training_eigenweight_names;
    int num_training_eigenweights;
    char** testing_image_names;
    int num_testing_images;
    int pixels_per_image;
    int width_in_pixels;
    int height_in_pixels;
} JobInfo;


typedef struct
{
    char tested_image_name[ STD_STR_LEN ];
    int test_image_index;
    char matched_image_name[ STD_STR_LEN ];
    int matched_image_index;
    double distance_between_images;
    char is_match_by_test;
    char is_actual_match;
} MatchResult;

/*==============================================================================
=======     GLOBAL VARIABLES     ===============================================
==============================================================================*/


/*==============================================================================
=======     FUNCTION PROTOTYPES     ============================================
==============================================================================*/
/**
FunctionName

A short description

@param

@return

@pre
-# 

@post
-# 

@detail @bAlgorithm
-# 

@exception

@code
@endcode
*/
int processCommandLineArguments( const int argc, char** argv,
                                 JobInfo& info );

int trainingMode( JobInfo& info );

int findSampleImageSize( int& image_width, int& image_height,
                         const char* file_name );

int allocateArrays( double**& face_data_matrix, double*& mean_face, const JobInfo& info );

int loadTrainingData( Matrix<double, Dynamic, Dynamic>& face_data, const JobInfo& info );

int readPgmToVector( VectorXd& image_vector, const char* file_name, const JobInfo& info );

int trainMeanFace( VectorXd& mean_face,  const Matrix<double, Dynamic, Dynamic>& face_data, const JobInfo& info );

int writeVectorToFile( const char* file_name, const VectorXd& data_vector );

int centerFaceData( Matrix<double, Dynamic, Dynamic>& centered_data, Matrix<double, Dynamic, Dynamic>& face_data, const VectorXd& mean_face, const JobInfo& info );

int findFaceCovariance( Matrix<double, Dynamic, Dynamic>& face_covariance,
                        const Matrix<double, Dynamic, Dynamic>& centered_data,
                        const JobInfo& info );

int findEigenValuesVectors( VectorXd& eigenvalues, MatrixXd& eigenvectors,
                            const MatrixXd& A_matrix, const JobInfo& info );

bool compareEigenvalIndexPairs( const pair< double, int >& one,
                                const pair< double, int >& other );

double getVectorMagnitude( const VectorXd the_vector );

int writePgmImage( const string& file_name, const VectorXd& image_data,
                   int image_width, int image_height, int image_shades );

VectorXd scaleImageToPgm( const VectorXd& image_data, const JobInfo& info );

int storeEigenData( const VectorXd& eigenvalues, const MatrixXd& eigenvectors );

int promptForNumEigens( const VectorXd& eigenvalues );

int storeEigenProjections( const int eigens_to_keep, const MatrixXd& centered_data, const MatrixXd& face_eigenvectors, const JobInfo& info );

int writeEigenweightFile( const char* file_name, const VectorXd& eigen_weights );

int testingMode( JobInfo& info );

int loadTrainingOutcomes( VectorXd& mean_face, VectorXd& eigenvalues, MatrixXd& eigenvectors, MatrixXd& training_eigenweights, const JobInfo& info );

int readDataVectorFromTxt( VectorXd& data_vector, const char* file_name );

int readMatrixFromTxt( MatrixXd& data_matrix, const char* file_name );

int loadTestingData( Matrix<double, Dynamic, Dynamic>& face_data, const JobInfo& info );

int centerTestData( MatrixXd& centered_data, MatrixXd& face_data, const VectorXd& mean_face, const JobInfo& info );

int projectDataInEigenspace( MatrixXd& eigenweights, const MatrixXd& data, const MatrixXd& eigenvectors );

VectorXd projectImageInEigenspace( const VectorXd& centered_image, const MatrixXd& eigenvectors  );

double promptForThreshold();

int matchAllImages( vector< MatchResult >& matching_results,
                 const MatrixXd& training_eigenweights,
                 const MatrixXd& testing_eigenweights,
                 const MatrixXd& eigenvalues,
                 const double threshold,
                 const int num_eigens_to_use,
                 const JobInfo& info );

int matchImage( MatchResult& result, const VectorXd& test_weights, const MatrixXd& training_weights );

int matchImage( MatchResult& result, const VectorXd& test_weights,
                const MatrixXd& training_weights, const VectorXd& eigenvalues,
                const int num_eigens_to_use, const double threshold,
                const JobInfo& info );

double computeMahalanobisDistance( const VectorXd& test_weights,
                                   const VectorXd& training_weights,
                                   const VectorXd& eigenvalues,
                                   const int num_eigens_to_use );

int outputDataSummary( vector< MatchResult >& results, double threshold, int num_eigens_to_use, bool explicit_summary );

char* stripFilePath( char* file_name );

/*==============================================================================
=======     MAIN FUNCTION     ==================================================
==============================================================================*/

/**
FunctionName

A short description

@param

@return

@pre
-# 

@post
-# 

@detail @bAlgorithm
-# 

@exception

@code
@endcode
*/
int main( int argc, char** argv )
{
    // variables
    int program_status = NO_ERRORS;
    int program_mode = TRAINING_MODE;
    JobInfo info;

    // process the command line arguments
    program_mode = processCommandLineArguments( argc, argv, info );

info.pixels_per_image = 2880; // 48 x 60 for hi-res samples

    // case: the program is running in training mode
    if( program_mode == TRAINING_MODE )
    {
        // indicate that the program is running in training mode
        cout << char( 0x0C )
             << "    =====================" << endl
             << "    |   TRAINING MODE   |" << endl
             << "    =====================" << endl << endl;

        // run program in training mode
        trainingMode( info );
    }
    // case: the program is running in "testing" mode
    else if( program_mode == TESTING_MODE )
    {
        // indicate that the program is running in training mode
        cout << char( 0x0C )
             << "       ====================" << endl
             << "       |   TESTING MODE   |" << endl
             << "       ====================" << endl << endl;

        testingMode( info );
    }    
    // case: bad command line arguments were used
    else
    {
        puts( "Bad command line arguments result in program termination.\n" );
    }

    // return a program operation status signal
    return program_status;
}


/*==============================================================================
=======     FUNCTION IMPLEMENTATIONS     =======================================
==============================================================================*/
int processCommandLineArguments( const int argc, char** argv,
                                 JobInfo& info )
{
    // variables
    int program_mode = INVALID;

    // case: a large enough number of command line arguments were given
    if( argc >= 3 )
    {
        // collect the numerical value of the first command line argument
        program_mode = atoi( argv[1] );

        // case: the given mode is not a valid one
        if( !( program_mode == TRAINING_MODE ||
               program_mode == TESTING_MODE    ) )
        {
            // set the mode to indicate an invalid mode
            program_mode = INVALID;
        }

        if( program_mode == TRAINING_MODE )
        {
            // set up the job information
            info.num_training_images = argc - 2;
            info.training_image_names = argv + 2;
        }
        else if( program_mode == TESTING_MODE )
        {
            // set up the job information
            info.training_image_names = argv + 2;
            info.num_training_images = 0;
            while( strstr( info.training_image_names[ info.num_training_images ], ".pgm" ) != NULL )
            {
                info.num_training_images++;
            }

            info.training_eigenweight_names = info.training_image_names + info.num_training_images;
            info.num_training_eigenweights = 0;
            while( strstr( info.training_eigenweight_names[ info.num_training_eigenweights ], ".txt" ) != NULL )
            {
                info.num_training_eigenweights++;
            }

            info.testing_image_names = info.training_eigenweight_names + info.num_training_eigenweights;
            info.num_testing_images = 0;
            while( ( info.num_testing_images < 1196 ) &&
                   ( strstr( info.testing_image_names[ info.num_testing_images ], ".pgm" ) != NULL ) )
            {
                info.num_testing_images++;
            }
        }
    }
    // case: too few arguments were used
    else
    {
        // give a stern message
        cout << "Something's wrong there guy... You need more arguments."
             << endl << endl;
    }

    // return the mode the program will be running in
    return program_mode;
}


int trainingMode( JobInfo& info )
{
    // variables
    int training_success = NO_ERRORS;

    VectorXd mean_face;
    VectorXd face_eigenvalues;
    Matrix< double, Dynamic, Dynamic > face_eigenvectors;
    Matrix< double, Dynamic, Dynamic > face_data;
    Matrix< double, Dynamic, Dynamic > centered_data; // A matrix
    Matrix< double, Dynamic, Dynamic > face_covariance;
    Matrix< double, Dynamic, Dynamic > eigen_faces;
    pair< VectorXd, MatrixXd > eigen_values_vectors;

    int eigens_to_keep = 1;

    info.pixels_per_image = findSampleImageSize( info.width_in_pixels,
                                                 info.height_in_pixels,
                                                 info.training_image_names[0] );

    puts( "Loading data...\n" );
    loadTrainingData( face_data, info );

    puts( "Training mean...\n" );
    trainMeanFace( mean_face, face_data, info );

    puts( "\"Centering\" faces...\n" );
    centerFaceData( centered_data, face_data, mean_face, info );

/*    puts( "Training covariance matrix...\n" );
    findFaceCovariance( face_covariance, centered_data, info );
 */

    // compute the Eigen values/vectors
    puts( "Finding Eigenvalues/Eigenvectors...\n" );
    findEigenValuesVectors( face_eigenvalues, face_eigenvectors,
                            centered_data, info );

    // create data/coefficients file
    puts( "Storing the found eigen-data...\n" );
    storeEigenData( face_eigenvalues, face_eigenvectors );

    // prompt user for the amount of information to keep
    eigens_to_keep = promptForNumEigens( face_eigenvalues );

    // create the reduced dimensionality eigenfaces, store them to a file
    puts( "Storing the training faces projected into eigenspace...\n" );
    storeEigenProjections( eigens_to_keep, centered_data, face_eigenvectors, info );


    // return a signal as to whether or not training was successful
    return training_success;
}


int findSampleImageSize( int& image_width, int& image_height,
                         const char* file_name )
{
    ifstream fin;
    string magic_number;
    int width = 0;
    int height = 0;

    fin.clear();
    fin.open( file_name );
    fin >> magic_number >> image_width >> image_height;
    fin.close();

    return ( image_width * image_height );
}


int loadTrainingData( Matrix<double, Dynamic, Dynamic>& face_data, const JobInfo& info )
{
    // variables
    int image_num = 0;
    VectorXd temp;

    face_data.resize( info.pixels_per_image, info.num_training_images );

    // for every image
    for( image_num = 0; image_num < info.num_training_images; image_num++ )
    {
        readPgmToVector( temp, info.training_image_names[image_num], info );
        face_data.col( image_num ) = temp;
    }


#if 0
writePgmImage( info.training_image_names[20], face_data.col( 20 ), info.width_in_pixels,
               info.height_in_pixels, 255 );
#endif

}


int readPgmToVector( VectorXd& image_vector, const char* file_name, const JobInfo& info )
{
    int read_success = NO_ERRORS;
    int pixel_num = 0;
    ifstream fin;
    string magic_number;
    int width;
    int height;
    int granularity;
    unsigned char pixel_byte;

    // open the file
    fin.clear();
    fin.open( file_name );

    // size the vector appropriately
    image_vector.resize( info.pixels_per_image );

    // eat the header
    fin >> magic_number >> width >> height >> granularity;
    fin.get(); // and an endline char

    // read in the data to a column vector
    for( pixel_num = 0; pixel_num < info.pixels_per_image; pixel_num++ )
    {
        // read in the byte, convert to float
        pixel_byte = fin.get();
        image_vector( pixel_num ) = (double) pixel_byte;
    }

    // close the file
    fin.close();

    return read_success;
}


int trainMeanFace( VectorXd& mean_face,  const Matrix<double, Dynamic, Dynamic>& face_data, const JobInfo& info )
{
    int mean_success = NO_ERRORS;
    int image_num = 0, pixel_num = 0;

    mean_face.resize( info.pixels_per_image );

    for( pixel_num = 0; pixel_num < info.pixels_per_image; pixel_num++ )
    {
        for( image_num = 0, mean_face( pixel_num ) = 0;
             image_num < info.num_training_images;
             image_num++ )
        {
            mean_face( pixel_num ) += face_data( pixel_num, image_num );
        }

        mean_face( pixel_num ) /= info.num_training_images;
    }


#if 1
writePgmImage( "mean_training_face.pgm", mean_face, info.width_in_pixels,
               info.height_in_pixels, 255 );

writeVectorToFile( "mean_training_face_data.txt", mean_face );
#endif

    return mean_success;
}


int writeVectorToFile( const char* file_name, const VectorXd& data_vector )
{
    int write_success = NO_ERRORS;
    ofstream fout;
    int i = 0;

    fout.clear();
    fout.open( file_name );
    fout << data_vector.size() << endl;
    fout<< data_vector << endl;

    return write_success;
}


int centerFaceData( Matrix<double, Dynamic, Dynamic>& centered_data, Matrix<double, Dynamic, Dynamic>& face_data, const VectorXd& mean_face, const JobInfo& info )
{
    int centering_success = NO_ERRORS;
    int image_number = 0, pixel_number = 0;

    centered_data.resize( face_data.rows(), face_data.cols() );

    for( image_number = 0; image_number < info.num_training_images; image_number++ )
    {
        for( pixel_number = 0; pixel_number < info.pixels_per_image; pixel_number++ )
        {
            centered_data( pixel_number, image_number ) = face_data( pixel_number, image_number ) - mean_face( pixel_number );
        }
    }


#if 0
writePgmImage( "test_center.pgm", centered_data.col( 5 ), info.width_in_pixels,
               info.height_in_pixels, 255 );
#endif

    return centering_success;
}


int findFaceCovariance( Matrix<double, Dynamic, Dynamic>& face_covariance,
                        const Matrix<double, Dynamic, Dynamic>& centered_data,
                        const JobInfo& info )
{
    int covariance_success = NO_ERRORS;
    double inverse_num_training_images = 1.0 / info.num_training_images;

    face_covariance.resize( info.pixels_per_image, info.pixels_per_image );

    face_covariance = inverse_num_training_images *
                      ( centered_data * centered_data.transpose().eval() );

    return covariance_success;
}


bool compareEigenvalIndexPairs( const pair< double, int >& one,
                                const pair< double, int >& other )
{
    // this makes for a high to low sort when used with std::sort
    return ( one.first > other.first );
}


double getVectorMagnitude( const VectorXd the_vector )
{
    double magnitude = 0.0;
    int i = 0;

    for( i = 0; i < the_vector.size(); i++ )
    {
        magnitude += ( the_vector( i ) * the_vector( i ) );
    }

    return sqrt( magnitude );
}


int writePgmImage( const string& file_name, const VectorXd& image_data, int image_width, int image_height, int image_shades )
{
    int write_success = NO_ERRORS;
    ofstream fout;
    int pixel_num = 0;
    int row_sizer = 0;
    int num_pixels = image_width * image_height;

    fout.clear();
    fout.open( file_name.c_str() );


    // write header with PGM "magic number" for binary type
    fout << "P5" << '\n' << image_width << ' '
         << image_height << '\n' << image_shades << '\n';

    // write the data
    for( pixel_num = 0; pixel_num < num_pixels; pixel_num++ )
    {
        fout << (unsigned char) ((unsigned int)image_data( pixel_num ) % (image_shades + 1));
    }

#if 0
    // write header with PGM "magic number" for ASCII type
    fout << "P2" << ' ' << image_width << ' '
         << image_height << ' ' << image_shades << '\n';

    // write the data
    for( pixel_num = 0; pixel_num < num_pixels; pixel_num++ )
    {
        fout << (unsigned int) image_data( pixel_num ) % image_shades;
        row_sizer++;
        if( row_sizer < image_width )
        {
            fout << ' ';
        }
        else
        {
            row_sizer = 0;
            fout << '\n';
        }
    }
#endif
    fout.close();

    return write_success;
}


int findEigenValuesVectors( VectorXd& eigenvalues, MatrixXd& eigenvectors, const MatrixXd& A_matrix, const JobInfo& info )
{
    EigenSolver<MatrixXd> eigen_solver;
    Matrix<double, Dynamic, Dynamic> At_A_matrix;   // A^t * A matrix
    Matrix<double, Dynamic, Dynamic> interim_eigenvectors;
    vector< pair< double, int> > eigen_indices;
    pair< double, int > temp;
    int i = 0;
    char buffer[100];

    // compute the A^t * A matrix to save on computations vs A * A^t
    At_A_matrix = A_matrix.transpose().eval() * A_matrix;

    // compute the eigen-stuff of the matrix
    eigen_solver.compute( At_A_matrix );

    // identify the strictly real eigen values and vectors
    for( i = 0; i < eigen_solver.eigenvalues().size(); i++ )
    {
        if( ( eigen_solver.eigenvalues().imag()( i ) == 0.0 ) )
        {
            temp.first = eigen_solver.eigenvalues().real()( i );
            temp.second = i;
            eigen_indices.push_back( temp );
        }
    }

    // sort the eigen stuff in descending order by eigen value
    sort( eigen_indices.begin(), eigen_indices.end(),
          compareEigenvalIndexPairs );

    // store the eigen-stuff in usable objects
    eigenvalues.resize( eigen_indices.size() );
    interim_eigenvectors.resize( A_matrix.cols(), eigen_indices.size() );
    eigenvectors.resize( A_matrix.rows(), eigen_indices.size() );
    for( i = 0; i < eigen_indices.size(); i++ )
    {
        eigenvalues( i ) = eigen_indices[i].first;
        interim_eigenvectors.col( i ) = eigen_solver.eigenvectors().col( eigen_indices[i].second ).real();
    }

    // get from Av to u for actual eigenvectors
    for( i = 0; i < eigenvectors.cols(); i++ )
    {
        // get the u vector
        eigenvectors.col( i ) = (A_matrix * interim_eigenvectors.col( i ));

        // make it a unit u vector
        eigenvectors.col( i ) /= getVectorMagnitude( eigenvectors.col( i ) );
    }

    for( i = 0; i < eigenvalues.size(); i++ )
    {
        sprintf( buffer, "eigenfaces/eigenface_%d.pgm", i );

        writePgmImage( buffer, scaleImageToPgm( eigenvectors.col( i ), info ), info.width_in_pixels,
               info.height_in_pixels, 255 );
    }

    return 0;
}


VectorXd scaleImageToPgm( const VectorXd& image_data, const JobInfo& info )
{
    VectorXd scaled_vector;
    double minimum_value = image_data( 0 );
    double maximum_value = image_data( 0 );
    int pixel_num = 0;

    scaled_vector.resize( image_data.size() );
    scaled_vector = image_data;

    // find the minimum and maximum values for scaling
    for( pixel_num = 0; pixel_num < image_data.size(); pixel_num++ )
    {
        if( image_data( pixel_num ) < minimum_value )
        {
            minimum_value = image_data( pixel_num );
        }

        if( image_data( pixel_num ) > maximum_value )
        {
            maximum_value = image_data( pixel_num );
        }
    }

    // scale the pixel values properly
    for( pixel_num = 0; pixel_num < image_data.size(); pixel_num++ )
    {
        scaled_vector( pixel_num ) = image_data( pixel_num ) - minimum_value;
    }
    maximum_value -= minimum_value;
    scaled_vector *= ( 255.0 / (maximum_value ));//- minimum_value ));

    return scaled_vector;
}


int storeEigenData( const VectorXd& eigenvalues, const MatrixXd& eigenvectors )
{
    int storage_success = NO_ERRORS;
    ofstream fout;
    int i = 0;
    int j = 0;

    fout.clear();
    fout.open( "EIGENVALUES.txt" );
    fout << eigenvalues.size() << endl;
    for( i = 0; i < eigenvalues.size(); i++ )
    {
        fout << eigenvalues( i ) << endl;
    }
    fout.close();

    fout.clear();
    fout.open( "EIGENVECTORS.txt" );
    fout << eigenvectors.rows() << ' ' << eigenvectors.cols() << endl;

    fout << eigenvectors << endl;

    fout.close();

    return storage_success;
}


int promptForNumEigens( const VectorXd& eigenvalues )
{
    int num_eigens_to_keep = 0;
    double eigenvalue_sum = 0;
    double kept_eigenvalue_sum = 0;
    char response;
    bool keep_prompting = true;
    int i = 0;

    for( i = 0; i < eigenvalues.size(); i++ )
    {
        eigenvalue_sum += eigenvalues( i );
    }

    while( keep_prompting )
    {
        cout << "How many eigen values (faces) should be kept? ("
             << eigenvalues.size() << " available): ";
        cin >> num_eigens_to_keep;
 // use 16 for ~80%

        for( kept_eigenvalue_sum = 0, i = 0; i < num_eigens_to_keep; i++ )
        {
            kept_eigenvalue_sum += eigenvalues( i );
        }

        cout << ( kept_eigenvalue_sum * 100 / eigenvalue_sum )
             << "% of the information will be kept. Is this acceptable? (y/n): ";
        cin >> response;

        if( response == 'y' )
        {
            keep_prompting = false;
        }
    }

    return num_eigens_to_keep;
}


int storeEigenProjections( const int eigens_to_keep, const MatrixXd& centered_data, const MatrixXd& eigenvectors, const JobInfo& info )
{
    int eigenprojection_success = NO_ERRORS;
    VectorXd projected_face;
    MatrixXd eigen_weights;
    char buffer[100];
    ofstream fout;
    int i = 0;
    int j = 0;

    // initial setup and resizing
    eigen_weights.resize( eigens_to_keep, centered_data.cols() );
    projected_face.resize( info.pixels_per_image );

    for( j = 0; j < centered_data.cols(); j++ )
    {
       // eigen_weights.col( j ) = projectImageInEigenspace( centered_data.col( j ), eigenvectors );

        for( i = 0; i < eigens_to_keep; i++ )
        {
            eigen_weights( i, j ) = eigenvectors.col( i ).transpose().eval() * centered_data.col( j );
        }

    }

    for( j = 0; j < centered_data.cols(); j++ )
    {
        projected_face *= 0;

        for( i = 0; i < eigens_to_keep; i++ )
        {
            projected_face += eigen_weights( i, j ) * eigenvectors.col( i );
        }

        sprintf( buffer, "training_eigenprojections/projected_face_%d_coefficients.txt", j );
        writeEigenweightFile( buffer, eigen_weights.col( j ) );

        sprintf( buffer, "training_eigenprojections/projected_face_%d.pgm", j );
        writePgmImage( buffer, scaleImageToPgm( projected_face, info ), info.width_in_pixels,
               info.height_in_pixels, 255 );
    }

    return eigenprojection_success;
}


int writeEigenweightFile( const char* file_name, const VectorXd& eigen_weights )
{
    int write_success = NO_ERRORS;
    ofstream fout;

    fout.clear();
    fout.open( file_name );

    fout << eigen_weights.size() << endl;
    fout << eigen_weights << endl;
    
    fout.close();

    return write_success;
}


int testingMode( JobInfo& info )
{
    int testing_success = NO_ERRORS;
    VectorXd mean_face;
    VectorXd eigenvalues;
    MatrixXd eigenvectors;
    MatrixXd testing_faces;
    MatrixXd centered_test_faces;
    MatrixXd training_eigenweights;
    MatrixXd testing_eigenweights;
    int num_eigens_to_use = 0;
    vector< MatchResult > matching_results;
    double threshold = 1.0;
    ofstream fout;
    int i = 0;
    int true_positive = 0, false_positive = 0, total_positive = 0, total_negative = 0;

    info.pixels_per_image = findSampleImageSize( info.width_in_pixels,
                                                 info.height_in_pixels,
                                                 info.training_image_names[0] );

    puts( "Loading training outcomes...\n" );
    loadTrainingOutcomes( mean_face, eigenvalues, eigenvectors, training_eigenweights, info );

    puts( "Projecting inputs onto eigenspace...\n" );
    loadTestingData( testing_faces, info );
    centerTestData( centered_test_faces, testing_faces, mean_face, info );
    projectDataInEigenspace( testing_eigenweights, centered_test_faces, eigenvectors );

    puts( "Matching inputs to training images...\n" );
    fout.clear()
    fout.open( "ROC.txt" );
    fout << "True positives, total positives, false negatives, total negatives" << endl;
    for( threshold = 0.0001, true_positive = 0, false_positive = 0, total_positive = 0, total_negative = 0;
           threshold < 0.004; threshold += 0.0005 )
    {
        num_eigens_to_use = 114; //promptForNumEigens( eigenvalues );
        // threshold = 1.5; // promptForThreshold();
        matchAllImages( matching_results, training_eigenweights, testing_eigenweights, eigenvalues, threshold, num_eigens_to_use, info );

        puts( "Outputting matching data...\n" );
        outputDataSummary( matching_results, threshold, num_eigens_to_use, false );

        for( i = 0; i < matching_results.size(); i++ );
        


    }

    return testing_success;
}


int loadTrainingOutcomes( VectorXd& mean_face, VectorXd& eigenvalues, MatrixXd& eigenvectors, MatrixXd& training_eigenweights, const JobInfo& info )
{
    int load_success = NO_ERRORS;
    int i = 0;
    VectorXd temp;

    readDataVectorFromTxt( mean_face, "mean_training_face_data.txt" );
    readDataVectorFromTxt( eigenvalues, "EIGENVALUES.txt" );
    readMatrixFromTxt( eigenvectors, "EIGENVECTORS.txt" );

cout << eigenvalues.size() << endl;

    training_eigenweights.resize( 1204, eigenvectors.cols() );
    for( i = 0; i < info.num_training_eigenweights; i++ )
    {
        readDataVectorFromTxt( temp, info.training_eigenweight_names[i] );
        training_eigenweights.col( i ) = temp;
    }

    return load_success;
}


int readDataVectorFromTxt( VectorXd& data_vector, const char* file_name )
{
    int read_success = NO_ERRORS;
    ifstream fin;
    int size;
    double value;
    int i;

    fin.clear();
    fin.open( file_name );

    if( fin.good() )
    {
        fin >> size;
        data_vector.resize( size );

        for( i = 0; i < size; i++ )
        {
            fin >> value;
            data_vector( i ) = value;
        }
    }

    return read_success;
}


int readMatrixFromTxt( MatrixXd& data_matrix, const char* file_name )
{
    int read_success = NO_ERRORS;
    ifstream fin;
    int rows, cols;
    double value;
    int i = 0, j = 0;

    fin.clear();
    fin.open( file_name );

    if( fin.good() )
    {
        fin >> rows >> cols;
        data_matrix.resize( rows, cols );

        for( i = 0; i < rows; i++ )
        {
            for( j = 0; j < cols; j++ )
            {
                fin >> value;
                data_matrix( i, j ) = value;
            }
        }
    }

    return read_success;
}


int loadTestingData( Matrix<double, Dynamic, Dynamic>& face_data, const JobInfo& info )
{
    // variables
    int image_num = 0;
    VectorXd temp;

    face_data.resize( info.pixels_per_image, info.num_testing_images );

    // for every image
    for( image_num = 0; image_num < info.num_testing_images; image_num++ )
    {
        readPgmToVector( temp, info.testing_image_names[image_num], info );
        face_data.col( image_num ) = temp;
    }


#if 1
writePgmImage( info.training_image_names[20], face_data.col( 20 ), info.width_in_pixels,
               info.height_in_pixels, 255 );
#endif
}


int centerTestData( MatrixXd& centered_data, MatrixXd& face_data, const VectorXd& mean_face, const JobInfo& info )
{
    int centering_success = NO_ERRORS;
    int image_number = 0, pixel_number = 0;

    centered_data.resize( face_data.rows(), face_data.cols() );

    for( image_number = 0; image_number < info.num_testing_images; image_number++ )
    {
        for( pixel_number = 0; pixel_number < info.pixels_per_image; pixel_number++ )
        {
            centered_data( pixel_number, image_number ) = face_data( pixel_number, image_number ) - mean_face( pixel_number );
        }
    }


#if 0

writePgmImage( "test_center.pgm", centered_data.col( 5 ), info.width_in_pixels,
               info.height_in_pixels, 255 );
#endif

    return centering_success;
}

int projectDataInEigenspace( MatrixXd& eigenweights, const MatrixXd& data, const MatrixXd& eigenvectors )
{
    int eigenprojection_success = NO_ERRORS;
    int j = 0;

    // initial setup and resizing
    eigenweights.resize( eigenvectors.cols(), data.cols() );

    for( j = 0; j < data.cols(); j++ )  // for each image
    {
        eigenweights.col( j ) = projectImageInEigenspace( data.col( j ), eigenvectors );

/*
        for( i = 0; i < eigenvectors.cols(); i++ ) // for each feature/weight
        {
            eigenweights( i, j ) = eigenvectors.col( i ).transpose().eval() * data.col( j );
        }
*/
    }

    return eigenprojection_success;
}


VectorXd projectImageInEigenspace( const VectorXd& centered_image, const MatrixXd& eigenvectors  )
{
    VectorXd weight_vector;
    int i = 0;

    weight_vector.resize( eigenvectors.cols() );

    for( i = 0; i < eigenvectors.cols(); i++ ) // for every eigenvector
    {
        weight_vector( i ) = eigenvectors.col( i ).transpose().eval() * centered_image;
    }

    return weight_vector;
}


double promptForThreshold()
{
    double threshold;

    cout << "Enter the threshold value to be used: ";
    cin >> threshold;

    return threshold;   
}


int matchAllImages( vector< MatchResult >& matching_results,
                 const MatrixXd& training_eigenweights,
                 const MatrixXd& testing_eigenweights,
                 const MatrixXd& eigenvalues,
                 const double threshold,
                 const int num_eigens_to_use,
                 const JobInfo& info )
{
    int matching_success;
    int i = 0;
    int j = 0;
    double temp_distance = 0;
    double distance = 0;
    MatchResult temp;

    for( i = 0; i < testing_eigenweights.cols(); i++ )
    {
        strcpy( temp.tested_image_name, stripFilePath( info.testing_image_names[i] ) );

        matchImage( temp, testing_eigenweights.col( i ), training_eigenweights, eigenvalues, num_eigens_to_use, threshold, info );

        matching_results.push_back( temp );
    }

    return matching_success;
}


int matchImage( MatchResult& result, const VectorXd& test_weights,
                const MatrixXd& training_weights, const VectorXd& eigenvalues,
                const int num_eigens_to_use, const double threshold,
                const JobInfo& info )
{
    int match_function_success = NO_ERRORS;
    int i = 0;
    double temp_distance = 2E64;
    int CMC_counter = 0;

    // check against all other sets of weights
    for( i = 0, result.distance_between_images = 2E64; i < training_weights.cols(); i++ )
    {
        // find the distance between the image and a training one
        temp_distance = computeMahalanobisDistance( test_weights, training_weights.col( i ), eigenvalues, num_eigens_to_use );

        // case: the distance is better than what was thought to be the previous best
        if( temp_distance < result.distance_between_images )
        {
            // update the index of the match and the new, lower distance
            result.distance_between_images = temp_distance;
            result.matched_image_index = i;
        }

        if( atoi(result.tested_image_name) == atoi( stripFilePath( info.training_image_names[ i ] ) )
            && temp_distance < threshold )
        {
            CMC_counter++;
        }

    }

cout << "CMC" << CMC_counter << endl;

    // indicate which image the test one is closest to
    strcpy( result.matched_image_name,
            stripFilePath( info.training_image_names[ result.matched_image_index ] ) );

    // case: the best distance could be considered a match
    if( result.distance_between_images < threshold )
    {
        // indicate that there was a match, get matching image's name
        result.is_match_by_test = MATCH;
    }
    // case: the distance does not indicate a match
    else
    {
        result.is_match_by_test = NO_MATCH;
    }

    // determine if the images are an actual match
    // NOTE: I expect the atoi to read just the first segment of the image name, up to the first underscore
    if( atoi( result.tested_image_name ) == atoi( result.matched_image_name ) )
    {
        result.is_actual_match = MATCH;

cout << result.tested_image_name << "  " << result.matched_image_name << "  " << result.is_actual_match << ' ' <<  MATCH << endl;

    }
    else
    {
        result.is_actual_match = NO_MATCH;
    }

    return match_function_success;
}


double computeMahalanobisDistance( const VectorXd& test_weights,
                                   const VectorXd& training_weights,
                                   const VectorXd& eigenvalues,
                                   const int num_eigens_to_use )
{
    double mahalanobis_distance = 0;
    double euclidean_component;
    int i = 0;

    for( i = 0; i < num_eigens_to_use; i++ )
    {
        euclidean_component = ( test_weights( i ) - training_weights( i ) );
        euclidean_component *= euclidean_component;
        mahalanobis_distance += euclidean_component / eigenvalues( i );
    }

    return mahalanobis_distance;
}


int outputDataSummary( vector< MatchResult >& results, double threshold, int num_eigens_to_use, bool explicit_summary )
{
    ofstream fout;
    int i = 0;


    if( explicit_summary )
    {
        fout.clear();
        fout.open( "RESULTS.txt" );

        fout << "Test Image Name,"
                "Matched Image Name,"
                "Distance Between Images,"
                "Is Match By Test,"
                "Is Actually A Match,"
             << endl;    

        for( i = 0; i < results.size(); i++ )
        {
            fout << results[i].tested_image_name << ", "
                 << results[i].matched_image_name << ", "
                 << results[i].distance_between_images << ", "
                 << results[i].is_match_by_test << ", "
                 << results[i].is_actual_match << endl;
        }
    }
}


char* stripFilePath( char* file_name )
{
    char* stripped_string = file_name;
    char* temp;

    temp = strstr( stripped_string, "/" );
    while( temp != NULL )
    {
        stripped_string = temp + 1;
        temp = strstr( stripped_string, "/" );
    }

    return stripped_string;
}



