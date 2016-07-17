/**
    @file PA02_Mandelbrot_static.cpp

    @author Terence Henriod

    Parallel Mandelbrot Computation: Static Load Distribution

    @brief This program produces an image representative of the
           canonical Mandelbrot Set using static work allocation. The time
           required for the computations is also recorded for comparison
           with parallel algorithms that conduct the same operation. 

    @version Original Code 1.00 (2/28/2014) - T. Henriod

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

/*==============================================================================
=======     HEADER FILES     ===================================================
==============================================================================*/
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include "my_pgm.h"
#include "my_stopwatch.h"


/*==============================================================================
=======     CONSTANTS / MACROS     =============================================
==============================================================================*/
#define true 1
#define false 0
#define kSuccess 0
#define kError 1
#define kStandardTag 0
#define kMaster 0
#define kMaxPixelIterations 255
#define kNameSize 40
#define kComplexMin -2
#define kComplexMax 2
#define kNotificationItems 2
#define kMandelTag 0


/*==============================================================================

=======     USER DEFINED TYPES     =============================================

==============================================================================*/

/**
@struct Complex

A structure for representing a complex number.

@var real        The real component of the complex number.
@var imaginary   The imaginary component of the complex number.
*/
typedef struct
{
  double real;
  double imaginary;
} Complex;


/*==============================================================================
=======     GLOBAL VARIABLES     ===============================================
==============================================================================*/

  // none

/*==============================================================================
=======     FUNCTION PROTOTYPES     ============================================
==============================================================================*/

/**
processCommandLineArguments

Checks the command line arguments for the correct count and validity.
The argument values are then copied to variables local to main.

@param argc               The number of arguments passed on the command line.
@param argv               An array of C-strings containing all of the arguments
                          passed on the command line.
@param image_parameters   A pointer to a struct that will contain the image
                          name, width, and height data
@param do_make_image      A reference to a variable that indicates if this
                          particular program run should create a Mandelbrot
                          image

@return arguments_are_valid   Used to indicate to the main program if
                              the agruments are acceptable.

@pre
-# None

@post
-# In the case that there was a problem with the command line arguments are
   bad, an error code is returned.
-# If the arguments are acceptable, then the information is stored in the
   appropriate variables in main.

@detail @bAlgorithm
-# The count of the arguments is checked for the correct number
-# If all of the above conditions are met, then the command line argument
   values are stored to the program. Otherwise, an error code is returned.

@code
@endcode
*/
int processCommandLineArguments( int argc, char** argv,
                                 PGMImageData* image_parameters,
                                 int* do_make_image );


/**
allocateImageMatrix

Handles the dynamic allocation of a two dimensional byte array for purposes
of constructing a grayscale image.

@param width    The specified width (or number of columns) of the image
@param height   The specified height the image should take (or the number of
                rows)

@return matrix   A pointer to the newly allocated image matrix

@pre
-# The specified dimensions are positive numbers

@post
-# If all allocation is successful, a pointer to the entire matrix is returned
-# Otherwise, a NULL pointer is returned

@detail @bAlgorithm
-# First, an array of pointers is allocated for each row of the image matrix
-# Then, memory for a byte for each column is allocated using each row pointer

@code
@endcode
*/
char** allocateImageMatrix( int width, int height );


/**
deallocateImageMatrix

Appropriately frees all of the memory for a given image matrix

@param matrix   The pointer to the entire matrix
@param height   The number of rows that need to be deallocated

@pre
-# The specified dimensions are positive numbers

@post
-# If all allocation is successful, a pointer to the entire matrix is returned
-# Otherwise, a NULL pointer is returned

@detail @bAlgorithm
-# First, an array of pointers is allocated for each row of the image matrix
-# Then, memory for a byte for each column is allocated using each row pointer

@code
@endcode
*/
void deallocateImageMatrix( char** matrix, int height );



















double staticMandelbrotCollect( PGMImageData* mandel_image );

void staticMandelbrotCompute( int processor_rank, int num_processors,
                              PGMImageData* mandel_image );


/**
computeMandelbrotImageRow

Accepts an array and a row number in order to compute the values for each
pixel in the row.

@param row            A pointer to the row where the new data will be placed
@param row_num        The number/index of the row currently being processed
@param image_width    Used both to indicate the length of the row and to scale
                      the starting complex value for the pixel computation
@param image_height   Used to scale the starting complex value for the
                      pixel computation

@pre
-# Parameter row points to a properly allocated row in an image matrix
-# The dimensional values must properly describe the row size to prevent
   segmentation faulting

@post
-# Parameter row will point to an array filled with computed Mandelbrot image
   pixel values

@detail @bAlgorithm
-# Simply iterates across the row, using the pixel coordinates to compute and
   store a Mandelbrot pixel value

@code
@endcode
*/
void computeMandelbrotImageRow( char* row, int row_num, int image_width,
                                int image_height );


/**
getComplexCoordinate

Accepts a complex number reference and prepares it to be the start of a pixel
calculation.

@param complex_coordinate   A reference to the complex coordinate to be
                            prepped for the computaion
@param row                  The row of the pixel to be computed
@param column               The column of the pixel to be computed
@param image_width          Used to scale the starting complex value for the
                            pixel computation
@param image_height         Used to scale the starting complex value for the
                            pixel computation

@pre
-# complex_coordinate must point to properly allocated memory
-# image_width and image_height must indicate the actual image dimensions

@post
-# A complex value will be prepared to initialize a Mandelbrot pixel
   calculation

@detail @bAlgorithm
-# Simply computes scaled starting complex coordinates for a given pixel

@code
@endcode
*/
void getComplexCoordinate( Complex* complex_coordinate, int row, int column,
                           int image_width, int image_height );

/**
calculatePixelValue

Performs the computations necessary to designate a color for a pixel in a
Mandelbrot set.

@param complex_coordinate   A complex coordinate used to prime the pixel
                            computation

@return iteration   The iteration number minus one is returned (the number of
                    iterations the calculation went through without being
                    stopped is how color is assigned)

@pre
-# The complex_coordinate must be an appropriate starting value

@post
-# A number corresponding to the pixel color is for a given coordinate 
   is returned

@detail @bAlgorithm
-# Iteratively refine the complex value for a given coordinate pair until the
   value is either found to be divergent ( >2 ) or the maximum allowed number
   of iterations has been reached
-# Track the count of iterations used
-# return the number of iterations - 1 (for byte data scaling purposes)

@code
@endcode
*/
char calculateMandelbrotPixel( Complex* complex_coordinate );


/**
reportResults

Reports the results of the Mandelbrot image computations in a format
intended for human consumption.

@param computation_time   The time that was required to perform the computation
                          by the program
@param width              The width of the created Mandelbrot image
@param height             The height of the created Mandelbrot image
@param num_processors     The number of processors utilized by the program
@param image_file_name    The name of the file that the image was stored
                          to, as a C-string
@pre
-# All computations have been completed

@post
-# The results of the program run will be put to the standard out

@detail @bAlgorithm
-# Simply diplays the necessary information

@code
@endcode
*/
void reportResults( double computation_time, PGMImageData* mandel_image,
                    int num_processors, int processor_rank );


/**
reportHostName

Simply a wrapper function for reporting the host machine's name. Handles the
case where hostname is not in the environment variables and available.

@pre
-# None, although this function is more helpful if the hostname is in the
   environment variables list.

@post
-# The host machine's name will either be displayed or the error will be
   reported.

@detail @bAlgorithm
-# A call to getenv() is made.
-# If any name is collected, it is reported.
-# If getenv() returns NULL, then the funciton reports that the program
   could not figure out it's environment name.

@code
@endcode
*/
void reportHostName();


/*==============================================================================
=======     MAIN FUNCTION     ==================================================
==============================================================================*/

/**
main

The main driver for the serial Mandelbrot set generation.

@param argc   The count of arguments the program was passed with from the
              command line.
@param argv   The list of arguments passed via the command line as c-strings.

@return program_success   A signal to the OS to indicate if the program
                          executed properly.

@pre
-# The program must be run in an MPI supportive environment.

@post
-# The program will accomplish its task of testing the message passing speed
   and report pertinent details.

@code
@endcode
*/
int main( int argc, char** argv )
{
  // variables
  int program_status = kError;
  int num_processors = 1;
  int processor_rank = 0;
  int do_make_image = true;
  double computation_time = 0;
  PGMImageData dummy;                     // stupid, I know
  PGMImageData* mandel_image = &dummy;

  // intialize MPI
  MPI_Init( &argc, &argv );

  // find the size of the MPI world
  MPI_Comm_size( MPI_COMM_WORLD, &num_processors );

  // find this processor's rank
  MPI_Comm_rank( MPI_COMM_WORLD, &processor_rank );

  // verify command line arguments
  program_status = processCommandLineArguments( argc, argv, mandel_image,
                                                &do_make_image );

  // case: command line arguments are good
  if( program_status == kSuccess )
  {
    // case: this is the first processor
    if( processor_rank == kMaster )
    {
      // allocate the image matrix
      mandel_image->data = allocateImageMatrix( mandel_image->width,
                                                mandel_image->height );

      // collect the data as other processors compute it
      computation_time = staticMandelbrotCollect( mandel_image );

      // case: an image should be made
      if( do_make_image )
      {
        // create the image
        createPGMimage( mandel_image );
      }

      // deconstruct the image matrix
      deallocateImageMatrix( mandel_image->data, mandel_image->height );
    }
    // case: this is a slave processor
    else
    {
      // perform the computations
      staticMandelbrotCompute( processor_rank, num_processors, mandel_image );
    }

    // report the results of the experiment
    reportResults( computation_time, mandel_image, num_processors,
                   processor_rank );
  }

  // finalize MPI
  MPI_Finalize();

  // return the program status code
  return program_status;
}


/*==============================================================================
=======     FUNCTION IMPLEMENTATIONS     =======================================
==============================================================================*/


int processCommandLineArguments( int argc, char** argv,
                                 PGMImageData* image_parameters,
                                 int* do_make_image )
{
  // variables
  int arguments_are_valid = kError;

  // case: the count of arguments is correct
  if( argc > 3 )
  {
    // the command line arguments are probably good TODO: actual error checking
    arguments_are_valid = kSuccess;

    // get the desired image name
    strcpy( image_parameters->name, argv[1] );

    // get the image dimensions
    image_parameters->width = atoi( argv[2] );
    image_parameters->height = atoi( argv[3] );

    // set the number of shades and data pointer
    image_parameters->data = NULL;
    image_parameters->num_shades = 0xFF;

    // case: there is a 5th command line argument
    if( argc > 4 )
    {
      // case: that argument dictates an image file should not be made
      if( strcmp( argv[4], "N" ) == 0 )
      {
        // update the value of the flag to reflect this
        *do_make_image = false;
      }
    }
  }

  // return the program status after processing the command line arguments
  return arguments_are_valid;
}


char** allocateImageMatrix( int width, int height )
{
  // variables
  char** matrix = NULL;
  int row_ndx = 0;
//  int col_ndx = 0;

  // allocate the set of row pointers
  matrix = (char**) calloc( height, sizeof( char* ) );

  // allocate the memory for each row
  for( row_ndx = 0; row_ndx < height; row_ndx++ )
  {
    // allocate the new row
    matrix[row_ndx] = (char*) calloc( width, sizeof( char ) );
  }

  // return the main pointer
  return matrix;
}


void deallocateImageMatrix( char** matrix, int height )
{
  // variables
  int row_ndx = 0;

  // deallocate the memory for each row
  for( row_ndx = 0; row_ndx < height; row_ndx++ )
  {
    // free up each row
    free( matrix[row_ndx] );
    matrix[row_ndx] = NULL;
  }

  // deallocate the set of row pointers
  free( matrix );
  matrix = NULL;

  // no return - void
}


double staticMandelbrotCollect( PGMImageData* mandel_image )
{
  // variables
  int rows_remaining = mandel_image->height;
  int row_number = 0;
  double elapsed_time = 0;
  MPI_Status message_status;

  // synchronize the processes
  MPI_Barrier( MPI_COMM_WORLD );

  // start the stopwatch
  stopwatch( 's' );

  // collect all the rows
  while( rows_remaining > 0 )
  {
    // collect the incoming row complete notification
    MPI_Recv( &row_number, 1, MPI_INT,
              MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &message_status );


    MPI_Recv( (mandel_image->data)[row_number], mandel_image->width, MPI_CHAR,
              message_status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
              &message_status );
/*
    // receive the data pixel by pixel to prevent message truncation
    for( col_ndx = 0; col_ndx < mandel_image->width; col_ndx++ )
    {
      // collect the row data
      MPI_Recv( &pixel, 1, MPI_CHAR,
                message_status.MPI_SOURCE, MPI_ANY_TAG,
                MPI_COMM_WORLD, &message_status );

      // store the pixel
      mandel_image->data[row_number][col_ndx] = pixel;
    }

*/


    // count the row as completed
    rows_remaining--;
  }

  // stop the stopwatch
  elapsed_time = stopwatch( 'x' );

  // return the elapsed time
  return elapsed_time;
}

void staticMandelbrotCompute( int processor_rank, int num_processors,
                              PGMImageData* mandel_image )
{
  // variables
  int current_row = 0;
  int row_increment = (num_processors - 1);
  char* mandelBuffer = NULL;

  // allocate the computation buffer
  mandelBuffer = (char*) calloc( mandel_image->width, sizeof( char ) );

  // synchronize the processes
  MPI_Barrier( MPI_COMM_WORLD );

  // the master will start the stopwatch at this point

  // compute all the necessary rows
  for( current_row = processor_rank - 1; current_row < mandel_image->height;
       current_row += row_increment )
  {
    // compute the Mandelbrot row values
    computeMandelbrotImageRow( mandelBuffer, current_row, mandel_image->width,
                               mandel_image->height );

    // send the row-complete message
    MPI_Send( &current_row, 1, MPI_INT, kMaster,
              kMandelTag, MPI_COMM_WORLD );


    MPI_Send( mandelBuffer, mandel_image->width, MPI_CHAR, kMaster,
              kMandelTag, MPI_COMM_WORLD );
/*
    // send the Mandelbrot row pixel by pixel to prevent truncation
    for( col_ndx = 0; col_ndx < mandel_image->width; col_ndx++ )
    {
      // send the pixel
      MPI_Send( &(mandelBuffer[col_ndx]), 1, MPI_CHAR, kMaster,
                kMandelTag, MPI_COMM_WORLD );
    }
*/
  }

  // deallocate the mandelBuffer
  free( mandelBuffer );

  // no return - void
}


void computeMandelbrotImageRow( char* row, int row_num, int image_width,
                                int image_height )
{
  // variables
  int col_ndx = 0;
  Complex coordinate;

  // compute a Mandelbrot pixel for every element of the array
  for( col_ndx = 0; col_ndx < image_width; col_ndx++ )
  {
    // get a scaled complex coordinate to perform the computation
    getComplexCoordinate( &coordinate, row_num, col_ndx, image_width,
                          image_height );

    // compute and store the pixel value
    row[col_ndx] = calculateMandelbrotPixel( &coordinate );
  }

  // no return - void
}

void getComplexCoordinate( Complex* complex_coordinate, int row, int column,
                           int image_width, int image_height )
{
  // variables
  static double real_scale = 0;
  static double imaginary_scale = 0;

  // compute the real scale value
  real_scale = ( kComplexMax - kComplexMin ) / (double) image_width;

  // compute the imaginary scale value
  imaginary_scale = ( kComplexMax - kComplexMin ) / (double) image_width;

  // compute the real component of the complex coordinate
  complex_coordinate->real = kComplexMin + ( (double) row * real_scale );

  // compute the imaginary component of the complex coordinate
  complex_coordinate->imaginary = kComplexMin +
                                  ( (double) column * real_scale );

  // no return - void
}

char calculateMandelbrotPixel( Complex* complex_coordinate )
{
  // variables
  int iteration = 0;
  double temp = 0;
  double length_squared = 0;
  Complex z;
    z.real = 0;
    z.imaginary = 0;

  // iteratively compute the pixel value until it is either seen that the 
  while( ( length_squared < 4.0 ) /* instead of a sqare root calculation*/ &&
         ( iteration <= kMaxPixelIterations ) )
  {
    // compute new z_real value (a^2 - b^2 + c)
    temp = (z.real * z.real) - (z.imaginary * z.imaginary);
    temp += complex_coordinate->real;

    // compute z_imaginary (2a_i * b_i + c_i), using the previous z_real
    z.imaginary = (2 * (z.real * z.imaginary)) + complex_coordinate->imaginary;

    // update the z_real value to the new one
    z.real = temp;

    // compute the magnitude of z^bar in the complex plane
    length_squared = (z.real * z.real) + (z.imaginary * z.imaginary);

    // update the iteration number
    iteration++;
  }

  // return the count of the iterations that were performed
  return ( iteration - 1 );
}


void reportResults( double computation_time, PGMImageData* mandel_image,
                    int num_processors, int processor_rank )
{
  // variables
    // none

  // print computer readabel output
  // case: this processor is the master
  if( processor_rank == kMaster )
  {
    // print the data in a computer accessible format
    printf( "%d,%d,%6.6F\r\n", mandel_image->width, num_processors,
            computation_time );
  }

  // no return - void
}


void reportHostName()
{
  // variables / get the host name
  const char* hostname = getenv( "HOSTNAME" );

  // case: the host name was accessed
  if( hostname != NULL )
  {
    // report the hostname
    printf( /*"I am running on host: */ "%s\n", hostname );
  }
  // case: the host name could not be found
  else
  {
    // report the lack of a host name
    printf( "I don't know where I am! Help!\n" );
  }

  // no return - void
}


