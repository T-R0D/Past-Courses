
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "mandelbrot.hpp"


__global__
void
NaiveMandelbrotKernel(
  unsigned* pixelValues,
  unsigned imageDimension,
  unsigned maxIterations) {

  unsigned totalThreads = gridDim.x * blockDim.x;
  unsigned numPixels = imageDimension * imageDimension;

  for (unsigned globalIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
       globalIndex < numPixels;
       globalIndex += totalThreads) {

    unsigned row = globalIndex / imageDimension;
    unsigned col = globalIndex % imageDimension;

    ComplexValue complexCoordinates = DeviceComputeComplexCoordinates(
      row,
      col,
      imageDimension,
      imageDimension,
      2,
      -2
    );

    pixelValues[globalIndex] = DeviceCalculateMandelbrotPixel(
      complexCoordinates,
      maxIterations
    );
  }
}

__device__
ComplexValue
DeviceComputeComplexCoordinates(
  const unsigned row,
  const unsigned column,
  const unsigned imageWidth,
  const unsigned imageHeight,
  const int complexMax,
  const int complexMin) {

  ComplexValue complexCoordinate = {0.0, 0.0};
  double realScale = 0;      // these were static variables in the old code
  double imaginaryScale = 0; // not sure why...

  realScale = (complexMax - complexMin) / (double) imageWidth;
  imaginaryScale = (complexMax - complexMin) / (double) imageWidth;

  complexCoordinate.real = complexMin + ((double) row * realScale );
  complexCoordinate.imaginary = complexMin + ((double) column * realScale );

  return complexCoordinate;
}

__device__
unsigned
DeviceCalculateMandelbrotPixel(
  const ComplexValue& complexCoordinate,
  const unsigned maxIterations) {

  int iteration = 0;
  double temp = 0;
  double length_squared = 0;
  ComplexValue z = {0.0, 0.0};

  while((length_squared < 4.0 ) /* instead of a sqare root calculation*/ &&
        (iteration <= maxIterations )) {
    // compute new z_real value (a^2 - b^2 + c)
    temp = (z.real * z.real) - (z.imaginary * z.imaginary);
    temp += complexCoordinate.real;

    // compute z_imaginary (2a_i * b_i + c_i), using the previous z_real
    z.imaginary = (2 * (z.real * z.imaginary)) + complexCoordinate.imaginary;

    z.real = temp;

    length_squared = (z.real * z.real) + (z.imaginary * z.imaginary);
    iteration++;
  }

  return iteration - 1;
}