# Which compiler to use
CC = g++ -std=c++11
# Any necessary compilation flags
CFLAGS = -I ~/cpp_libs/Eigen_lib
# Listing any dependencies (headers) for the project
DEPS = KalmanFilter.hpp

# This is a target. Targets get executed if any of their
# dependencies (listed after the ':') change
KalmanFilter: KalmanFilter_main.cpp KalmanFilter.cpp
	$(CC) -o KalmanFilter KalmanFilter_main.cpp KalmanFilter.cpp $(CFLAGS)
