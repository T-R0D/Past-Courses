/**
    @file WeightedGraph.h

    @author Terence Henriod

    Lab 11: WeightedGraph ADT

    @brief Class declarations for the WeighedGraph ADT and the Vertex inner
           class. Utilizes both an adjacency matrix and a path matrix.


    @version Original Code 1.00 (11/14/2013) - T. Henriod
*/


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   PREPROCESSOR DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#ifndef WEIGHTEDGRAPH_H
#define WEIGHTEDGRAPH_H


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include <stdexcept>
#include <iostream>
#include <climits>    // For INT_MAX
#include <string>     // Used for labels
using namespace std;


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   GLOBAL CONSTANTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

const string UNLABELED = "UNLABELED";


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
                   CLASS DEFINITION(S)
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
@class WeighedGraph

An implementation of a weighted graph that utilizes an adjacency matrix and a
path matrix. Supports graph coloring, although it does not support any kind of
graph coloring algorithms other than checking for a valid color. Contains an
inner vertex class.

@variable maxSize           The current maximum size capacity of the graph (in
                            terms of number of vertices).
@variable size              The current size (number of vertices) of the graph.
@variable vertexList        An array containing the vertices currently contained
                            in the graph.
@variable adjacencyMatrix   A 1-D array to be mapped as a 2-D array. In this
                            implementation, the 1-D array is blocked out in
                            terms of the maximum row size, and only the required
                            portions of each block are used. The helper
                            functions are designed to accomodate for this non-
                            contiguous array use. Each valid entry pertains to
                            an edge weight between two vertices, based on how
                            the particular index was mapped to.
@variable pathMatrix        This array is mapped in a similar manner to the
                            adjacency matrix, except it contains shortest path
                            (least cost) data.
*/
class WeightedGraph
{
 public:
 /*---   Class Specific Constants   ---*/
  static const int MAX_GRAPH_SIZE = 10; // Default number of vertices
  static const int INFINITE_EDGE_WT = INT_MAX;  // "Weight" of a missing edge
  static const int NOT_FOUND = -1;
  static const int LOOP_COST = 0;
  static const char DEFAULT_COLOR = 'R';

 /*---   Forward Declaration of Inner Class   ---*/
  class Vertex;

 /*---   Constructor(s) / Destructor   ---*/
  WeightedGraph( int maxNumber = MAX_GRAPH_SIZE );
  WeightedGraph( const WeightedGraph& other );
  WeightedGraph& operator=( const WeightedGraph& other );
  ~WeightedGraph();

 /*---   Mutators   ---*/
  void computePaths();
  void insertVertex( const Vertex& newVertex ) throw ( logic_error );
  void insertEdge( const string& v1, const string& v2, const int wt )
                   throw ( logic_error );
  bool retrieveVertex( const string& v, Vertex& vData ) const;
  bool getEdgeWeight( const string& v1, const string& v2, int& wt )
                      const throw ( logic_error );
  void removeVertex( const string& v ) throw ( logic_error );
  void removeEdge( const string& v1, const string& v2 )
                   throw ( logic_error );
  void clear();

 /*---   Accessors   ---*/
  bool areAllEven() const;
  bool hasProperColoring() const;
  bool isEmpty() const;
  bool isFull() const;
  void showShortestPaths();
  void showStructure() const;


 /*---   Public Data Members   ---*/
  /**
  @class Vertex

  A fundamental elemental element of a graph. Contains data pertaining to a
  label and color of a vertex contained in a graph.

  @variable label   The label of the vertex. Used as the key of the vertex.
  @variable color   The color the vertex currently has.
  */
  class Vertex
  {
   public:
   /*---   Mutators   ---*/
    void setColor(char newColor) { color = newColor; }
    void setLabel( const string& newLabel )
      { label = newLabel; }

   /*---   Accessors   ---*/
    char getColor() const { return color; }
    string getLabel() const { return label; }

   private:
    /*---   Private Data Members   ---*/
    string label;
    char color;
  };


 private:
 /*---   Helpers   ---*/
  // Mutator Helpers
  void setEdge( const int row, const int col, const int wt);
  void setPath( const int row, const int col, const int cost );

  // Accessor Helpers
  int getEdge( const int row, const int col ) const;
  int getIndex( const string& v ) const;
  int getPath( const int row, const int col ) const;


 /*---   Private Data Members   ---*/
    int maxSize;
    int size;
    Vertex* vertexList;
    int* adjacencyMatrix;
    int* pathMatrix;
};


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   TERMINATING PREPROCESSOR DIRECTIVES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#endif		// #ifndef WEIGHTEDGRAPH_H

