# Which compiler to use
CC = g++ -std=c++11
# Any necessary compilation flags
CFLAGS = -c
# Listing any dependencies (headers) for the project
DEPS = target_field.hpp sim_node_network.hpp sim_node.hpp

# This is a target. Targets get executed if any of their
# dependencies (listed after the ':') change
all: consensus

consensus: main.o sim_node_network.o sim_node.o
	$(CC) main.o sim_node_network.o sim_node.o -o consensus

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

sim_node_network.o: sim_node_network.cpp
	$(CC) $(CFLAGS) sim_node_network.cpp

sim_node.o: sim_node.cpp
	$(CC) $(CFLAGS) sim_node.cpp

clean:
	rm -rf *o consensus