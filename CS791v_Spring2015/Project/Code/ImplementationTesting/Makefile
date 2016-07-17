# Which compiler to use
CC = g++
# Any necessary compilation flags
CFLAGS = -c #-std=c++11
# Listing any dependencies (headers) for the project
DEPS = target_field.hpp sim_node_network.hpp sim_node.hpp
# This is a target. Targets get executed if any of their
# dependencies (listed after the ':') change
all: apsp

apsp: main.o
	$(CC) main.o -o apsp

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

clean:
	rm *.o apsp
