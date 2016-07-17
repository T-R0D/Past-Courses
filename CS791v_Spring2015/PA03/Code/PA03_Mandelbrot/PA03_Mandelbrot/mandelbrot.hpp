/**
 *
 */
#ifndef _MANDELBROT_HPP_
#define _MANDELBROT_HPP_ 1

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "complex_val.hpp"

/**
 * Computes the pixels for a Mandelbrot set the naive way: by striding across
 * an array's indices and placing an appropriately generated pixel value.
 */
__global__
void
NaiveMandelbrotKernel(
  unsigned* pixelValues,
  unsigned imageDimension,
  unsigned maxIterations);

__device__
ComplexValue
DeviceComputeComplexCoordinates(
  const unsigned row,
  const unsigned column,
  const unsigned imageWidth,
  const unsigned imageHeight,
  const int complexMax,
  const int complexMin);

__device__
unsigned
DeviceCalculateMandelbrotPixel(
  const ComplexValue& complexCoordinate,
  const unsigned maxIterations);

#endif //_MANDELBROT_HPP_