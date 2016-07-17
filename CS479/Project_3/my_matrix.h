#ifndef __MY_MATRIX_H__
#define __MY_MATRIX_H__


#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

/*==============================================================================
=======     CONSTANTS / MACROS     =============================================
==============================================================================*/
#define true 1
#define false 0

/**
@struct Matrix

A type that packages basic matrix information.

@var data     The pointer to the matrix' data array.
@var i_size   The height or number of rows of the matrix. This is typically
              labeled as I in mathematics.
@var j_size   The width or number of columns of the matrix. This is typically
              labeled as J in mathematics.
*/
typedef struct
{
  double* data;
  unsigned int i_size;
  unsigned int j_size;
} Matrix;


/**
@struct MatrixRows

A type that packages basic matrix information.

@var data                The pointer to the matrix' data array.
@var i_size              The index of the first row included in this block
                         of rows.
@var j_size              The width or number of columns of the matrix. This
                         is typically labeled as J in mathematics.
@var num_rows_included   The number of rows included in this block.
*/
typedef struct
{
  double* data;
  unsigned int i;
  unsigned int j_size;
  unsigned int num_rows_included;
} MatrixRows;


/*==============================================================================
=======     USER DEFINED TYPES     =============================================
==============================================================================*/



/*==============================================================================
=======     GLOBAL VARIABLES     ===============================================
==============================================================================*/
  // none


/*==============================================================================
=======     FUNCTION PROTOTYPES     ============================================
==============================================================================*/
/**
initializeMatrix

Allocates the memory for the specified for the matrix and updates the given
Matrix struct's members appropriately. The memory is initialized with zeroed
values.

@param new_matrix   The new Matrix to be initialized.
@param width        The width or number of columns the matrix will have.
@param height       The height or number of rows the matrix will have.

@pre
-# The given Matrix struct should not have any previously associated
   dynamic memory allocated, otherwise a memory leak will result.

@post
-# The given Matrix will be given memory that is zeroed of the
   given dimensions.

@detail @bAlgorithm
-# Dynamic memory is first allocated for the pointers to each of the rows.
-# Dynamic memory is then allocated for each row of integers.

@code
@endcode
*/
Matrix* createMatrix( unsigned int height, unsigned int width );


double getMatrixElement( Matrix* matrix, unsigned int i, unsigned int j );

void setMatrixElement( Matrix* matrix, unsigned int i, unsigned int j,
                       double new_value );

/**
deconstructMatrix

Returns any dynamic memory associated with the given matrix and zeroes its
dimensions.

@param matrix   The matrix to be destroyed.

@pre
-# None.

@post
-# The given Matrix will have no dynamic memory associated with it, and its
   dimensions will be zeroed.

@detail @bAlgorithm
-# First, rows are de-allocated.
-# Next, the array of pointers to the rows is de-allocated.
-# Finally, the dimension size members are set to 0.

@code
@endcode
*/
void destroyMatrix( Matrix** matrix );

int readMatrixFromFile( Matrix** matrix, char* file_name );

int writeMatrixToFile( Matrix* matrix, char* file_name );

/**
matrixRandomFill

Fills a given matrix with random (integer) values that are less than 10.

@param matrix   The matrix to be filled with random values.

@pre
-# This function only requires that a matrix struct be passed in.

@post
-# The given Matrix will be given random values for each cell.

@detail @bAlgorithm
-# Each cell is visited and filled with a random value 0 <= x < 10.

@code
@endcode
*/
void matrixRandomFill( Matrix* matrix );


/**
multiplyMatrices

Multiplies two matrices A and B and stores the result in C. The dimensions of
A and B are verified by assertion, so this is not user friendly code.

@param matrix_C   The result matrix, passed by reference.
@param matrix_A   The left operand matrix of the multiplication.
@param matrix_B   The right operand matrix of the multiplication.

@return operation_success   Indicates the success of conducting the matrix
                            multiplication.

@pre
-# Matrices A and B should be properly allocated and have matching dimensions,
   that is the number of rows in A must match the number of columns in B.

@post
-# Matrix C will be deconstructed and reallocated in order to contain the
   result of the multiplication.

@detail @bAlgorithm
-# An assertion check is performed to ensure that matrices A and B are
   compatible for multiplication.
-# Result matrix C is deconstructed and then re-allocated to the
   appropriate dimensions.
-# Every row of A is multiplied with every column of B, as is standard. The
   results are stored in C.

@code
@endcode
*/
int multiplyMatrices( Matrix* matrix_C, Matrix* matrix_A, Matrix* matrix_B );


int partialMatrixMultiply( Matrix* matrix_C, Matrix* matrix_A,
                           Matrix* matrix_B, int start_row, int last_row );


/**
printMatrix

Displays the values stored in the matrix in the terminal.

@param matrix   The matrix whose contents are to be diplayed.

@pre
-# None.

@post
-# The given Matrix' contents will be displayed in the terminal, row by row.

@detail @bAlgorithm
-# Each cell is visited, and it's contents are displayed.
-# Each row of the matrix is printed on one row of the screen.

@code
@endcode
*/
void printMatrix( Matrix* matrix );


/*==============================================================================
=======     FUNCTION IMPLEMENTATIONS     =======================================
==============================================================================*/

Matrix* createMatrix( unsigned int height, unsigned int width )
{
  // variables
  Matrix* new_matrix = NULL;
  unsigned int matrix_size = width * height;

  // allocate a new matrix
  new_matrix = (Matrix*) malloc( sizeof( Matrix ) );

  // set the matrix dimensions
  new_matrix->i_size = height;
  new_matrix->j_size = width;

  // allocate the new data array
  new_matrix->data = (double*) calloc( matrix_size, sizeof( double ) );

  // return the pointer to the new matrix
  return new_matrix;
}


double getMatrixElement( Matrix* matrix, unsigned int i, unsigned int j )
{
  // assert preconditions
  assert( ( 0 <= i ) && ( i < matrix->i_size ) );
  assert( ( 0 <= j ) && ( j < matrix->j_size ) );

  // variables
  unsigned int element_position = ( i * matrix->j_size ) + j;

  // return the appropriate element
  return matrix->data[element_position];
}


void setMatrixElement( Matrix* matrix, unsigned int i, unsigned int j,
                       double new_value )
{
  // assert preconditions
  assert( ( 0 <= i ) && ( i < matrix->i_size ) );
  assert( ( 0 <= j ) && ( j < matrix->j_size ) );

  // variables
  unsigned int element_position = ( i * matrix->j_size ) + j;

  // set the appropriate element
  matrix->data[element_position] = new_value;

  // return the matrix by reference
}


void destroyMatrix( Matrix** matrix )
{
  // case: a NULL pointer was not passed to the function
  if( matrix != NULL )
  {
    // case: the matrix was previously allocated
    if( (*matrix) != NULL )
    {
      // case: the matrix had a data array previously allocated
      if( (*matrix)->data != NULL )
      {
        // free the matrix data
        free( (*matrix)->data );
        (*matrix)->data = NULL;
      }

      // free the matrix pointer
      free( *matrix );
      (*matrix) = NULL;
    }
  }

  // return the matrix by reference
}


int readMatrixFromFile( Matrix** matrix, char* file_name )
{
  // variables
  int file_read_successful = false;
  FILE* file = NULL;
  int height = -1;
  int width = -1;
  int i = 0;
  int j = 0;
  double temp;

  // open the file for reading
  file = fopen( file_name, "r" );

  // case: file opening was successful
  if( file != NULL )
  {
    // read the dimensions of the matrix
    fscanf( file, "%d %d", &height, &width );

    // allocate the new matrix
    *matrix = createMatrix( width, height );

    // read in all the rows of the matrix
    for( i = 0; i < height; i++ )
    {
      // read in each element
      for( j = 0; j < width; j++ )
      {
        // perform the read
        fscanf( file, "%f", &temp );
        setMatrixElement( *matrix, i, j, temp );
      }
    }

    // close the file
    fclose( file );

    // indicate that the file opened suessfully
    file_read_successful = true;
  }

  // return the success of the file read
  return file_read_successful;
}


int writeMatrixToFile( Matrix* matrix, char* file_name )
{
  // variables
  FILE* file = NULL;
  int i = 0;
  int j = 0;

  // open the file for writing
  file = fopen( file_name, "a" );

  // write the matrix dimensions
  fprintf( file, "%d %d\r\n", matrix->i_size, matrix->j_size );

  // visit every row of the matrix data
  for( i = 0; i < matrix->i_size; i++ )
  {
    // print every element of the row
    for( j = 0; j < matrix->j_size; j++ )
    {
      // write the element
      fprintf( file, " %-4.1F", getMatrixElement( matrix, i, j ) );
    }

    // write an endline
    fprintf( file, "\r\n" );
  }

  // put some space at the end of the file
  fprintf( file, "\r\n\r\n" );

  // close the file
  fclose( file );
  file = NULL;

  // return writing success
  return true;  // TODO: implement error checking
}


void matrixRandomFill( Matrix* matrix )
{
  // variables
  unsigned int i = 0;
  unsigned int j = 0;

  // visit every row
  for( i = 0; i < matrix->i_size; i++ )
  {
    // visit every element of the row
    for( j = 0; j < matrix->j_size; j++ )
    {
      // fill the cell with a random number less than 10
      setMatrixElement( matrix, i, j, (double) ( rand() % 5 ) );
    }
  }

  // return the matrix by reference
}


int multiplyMatrices( Matrix* matrix_C, Matrix* matrix_A, Matrix* matrix_B )
{
  // assert pre-conditions
  assert( matrix_A->i_size == matrix_B->j_size );

  // variables
  int operation_success = true;  // TODO: various error handling
  unsigned int i = 0;
  unsigned int j = 0;
  unsigned int k = 0;
  double intermediate_sum = 0;

  // prepare the result matrix
  destroyMatrix( &matrix_C );
  matrix_C = createMatrix( matrix_A->i_size, matrix_B->j_size );

  // visit every row of A being multiplied
  for( i = 0; i < matrix_A->i_size; i++ )
  {
    // visit every column of B
    for( j = 0; j < matrix_B->j_size; j++ )
    {
      // sum the intermediate multiplications
      for( intermediate_sum = 0, k = 0; k < matrix_A->i_size; k++ )
      {
        // add the multiplication to the intermediate result
        intermediate_sum += ( getMatrixElement( matrix_A, i, k ) *
                              getMatrixElement( matrix_B, k, j ) );
      }

      // store the result to the appropriate location in the result matrix
      setMatrixElement( matrix_C, i, j, intermediate_sum );
    }
  }

  // return matrix_C by reference, return the success of the operation
  return operation_success;
}


int partialMatrixMultiply( Matrix* matrix_C, Matrix* matrix_A,
                           Matrix* matrix_B, int start_row, int last_row )
{
  // assert pre-conditions
  assert( matrix_A->i_size == matrix_B->j_size );

  // variables
  int operation_success = true;  // TODO: various error handling
  unsigned int i = 0;
  unsigned int j = 0;
  unsigned int k = 0;
  double intermediate_sum = 0;

  // prepare the result matrix
  destroyMatrix( &matrix_C );
  matrix_C = createMatrix( matrix_A->i_size, matrix_B->j_size );

  // visit every row of A being multiplied
  for( i = start_row; i <= last_row; i++ )
  {
    // visit every column of B
    for( j = 0; j < matrix_B->j_size; j++ )
    {
      // sum the intermediate multiplications
      for( intermediate_sum = 0, k = 0; k < matrix_A->i_size; k++ )
      {
        // add the multiplication to the intermediate result
        intermediate_sum += ( getMatrixElement( matrix_A, i, k ) *
                              getMatrixElement( matrix_B, k, j ) );
      }

      // store the result to the appropriate location in the result matrix
      setMatrixElement( matrix_C, i, j, intermediate_sum );
    }
  }

  // return matrix_C by reference, return the success of the operation
  return operation_success;
}


void printMatrix( Matrix* matrix )
{
  // variables
  unsigned int i = 0;
  unsigned int j = 0;

  // visit every row
  for( i = 0; i < matrix->i_size; i++ )
  {
    // visit every element
    for( j = 0; j < matrix->j_size; j++ )
    {
      // print the element
      printf( " %-4.1F", getMatrixElement( matrix, i, j ) );
    }

    // move down to the next line
    printf( "\r\n" );
  }

  // no return - void
}

#endif // ifndef my_matrix_h

