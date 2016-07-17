#ifndef __MY_PGM_PPMHACK_H__
#define __MY_PGM_PPMHACK_H__
/**
    @file my_pgm.h

    @author Terence Henriod

    @brief Provides a functions for using .ppm files HACKED FROM PGM.    

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
#include <stdio.h>
#include <assert.h>


/*==============================================================================
=======     CONSTANTS / MACROS     =============================================
==============================================================================*/
#define false 0
#define true 1
#define kImageNameLength 40


/*==============================================================================
=======     USER DEFINED TYPES     =============================================
==============================================================================*/

/**
@struct ImageInfo
A structure containing information relevant to the image format.

@var name         The image name
@var width        The width of the image
@var height       The height of the image
@var num_shades   The number of shades to be represented in the image.
                  Generally, 256 will be enough for my purposes, hence the
                  byte sized type.
@var data         A 2-d byte array pointer for the image data
*/
typedef struct
{
  char name[kImageNameLength];
  int width;
  int height;
  int num_shades;
  char** data; 
} PGMImageData;

typedef struct
{
  unsigned char red;
  unsigned char green;
  unsigned char blue;
} RGBTriple;

/*==============================================================================
=======     FUNCTION PROTOTYPES     ============================================
==============================================================================*/

/**
createPGMimage

Takes a given matrix of byte data and exports that data to a .pgm image file.

@param file_name      The name the the created file will have, as a C-string
@param image_width    The width of the image/number of data columns
@param image_height   The height of the image/the number rows for the
                      image/data
@param max_shades     The shade range for the data
@param image_data     A pointer to the image data matrix

@return success   The success of the operation. Returns false if the image
                  cannot be created, true if it can.

@pre
-# Valid dimensions that match the data must be given
-# All data must be given in bytes
-# The number of shades should be a normal count (it will be decremented
   to enable storage as a byte)

@post
-# A binary .pgm file will be created

@detail @bAlgorithm
-# A new image file with the given name is created
-# A header for a .pgm file is written
-# The data from the given matrix to the image file is written
-# The file is then closed

@code
@endcode
*/
int createPGMimage( PGMImageData* image );


/**
writePGMheader

Writes the header for the pgm file

@param image_file     A file pointer to the newly created file
@param image_width    The width of the image/number of data columns
@param image_height   The height of the image/the number rows for the
                      image/data
@param max_shades     The shade range for the data


@pre
-# Valid dimensions that match the data must be given
-# All data must be given in bytes
-# The number of shades should be a normal count (it will be decremented
   to enable storage as a byte)

@post
-# A binary .pgm file header will be written

@detail @bAlgorithm
-# The "magic number" for a binary grayscale format is written
-# A comment is written for the file
-# The dimensions of the image are written
-# Finally, the number of possible shades are written

@code
@endcode
*/
void writePGMheader( FILE* image_file, PGMImageData* image );


RGBTriple shadeToColor( unsigned char shade_level );


unsigned char crappyLgF( unsigned char shade_level );



/*==============================================================================
=======     FUNCTION IMPLEMENTATIONS     =======================================
==============================================================================*/

int createPGMimage( PGMImageData* image )
{
  // variables
  int success = false;
  FILE* the_file = NULL;
  int pixel_x = 0;
  int pixel_y = 0;
  unsigned char pixel = 0;
  RGBTriple colors;

  // assert necessities for variables
  assert( image->num_shades <= 255 ); // byte max
  assert( image->num_shades > 0 );

  // create a new file
  the_file = fopen( image->name, "wb" );

  // case: the file opening was successful
  if( the_file != NULL )
  {
    // update the success status
    success = true;

    // write the file header
    writePGMheader( the_file, image );

    // write the image data
    // iterate through rows
    for( pixel_x = 0; pixel_x < image->height; pixel_x++ )
    {
      // iterate across the row
      for( pixel_y = 0; pixel_y < image->width; pixel_y++ )
      {
        // print the pixel value
        pixel = (unsigned char) image->data[pixel_y][pixel_x];
        colors = shadeToColor( pixel );







        fprintf( the_file, "%c%c%c", (unsigned char) colors.red,
                                     (unsigned char) colors.green,
                                     (unsigned char) colors.blue );
      }

      // NOTE: ASCII mode requires a '\n' to end the row,
      //       binary mode will create a black line if you do this
    }

  // close the file
  fclose( the_file );
  }

  // return the success state of the operation
  return success;
}

void writePGMheader( FILE* image_file, PGMImageData* image )
{
  // variables
    // none

  // print the pgm (Portable Gray Map) format
  // use P2 for ASCII mode (gives easy human readability, requires spaces)
  // use P5 for binary mode (gives better storage and processing)
  fprintf( image_file, "P6\n" );

  // write the file comment
  fprintf( image_file, "# Makin' my PPM HACK file\n" );

  // specify image dimensions
  fprintf( image_file, "%d %d\n", image->width, image->height );

  // print the max shade value (granularity) number
  fprintf( image_file, "%d\n", image->num_shades );

  // no return - void
}


RGBTriple shadeToColor( unsigned char shade_level )
{
  RGBTriple returnval;
    returnval.red = 0;
    returnval.green = 0;
    returnval.blue = 0;

  if( shade_level % 2 == 0 )
    returnval.red = 20 * crappyLgF( shade_level );
  else
      returnval.red = shade_level / 2;
  if( shade_level % 3 == 0 )
    returnval.green = 20 * crappyLgF( shade_level );
  else
      returnval.green = shade_level / 3;
  if( shade_level % 4 == 0 )
    returnval.blue = 20 * crappyLgF( shade_level );
  else
      returnval.blue = shade_level / 4;

  return returnval;
}


unsigned char crappyLgF( unsigned char shade_level )
{
  unsigned char result = 0;

  if( shade_level <= 1 )
  {
    result = 0;
  }
  else if( shade_level <= 2 )
  {
    result = 1;
  }
  else if( shade_level <= 4 )
  {
    result = 2;
  }
  else if( shade_level <= 8 )
  {
    result = 3;
  }
  else if( shade_level <= 16 )
  {
    result = 4;
  }
  else if( shade_level <= 32 )
  {
    result = 5;
  }
  else if( shade_level <= 64 )
  {
    result = 6;
  }
  else if( shade_level <= 128 )
  {
    result = 7;
  }
  else if( shade_level <= 256 )
  {
    result = 8;
  }
  else if( shade_level <= 512 )
  {
    result = 9;
  }

  return result;
}

#endif

