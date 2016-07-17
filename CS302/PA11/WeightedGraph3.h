//--------------------------------------------------------------------
//
//  Laboratory 13, In-lab Exercise 2                      wtgraph3.h
//
//  Class declaration for the adjacency matrix implementation of
//  the Weighted Graph ADT -- including vertex coloring
//
//--------------------------------------------------------------------

#include <climits>  // For INT_MAX
#include <new>
#include <stdexcept>

using namespace std;

const int defMaxGraphSize = 10,      // Default number of vertices
          vertexLabelLength = 11,    // Length of a vertex label
          infiniteEdgeWt = INT_MAX;   // "Weight" of a missing edge

//--------------------------------------------------------------------

class Vertex
{
  public:

    char label [vertexLabelLength];   // Vertex label
    char color;                       // Vertex color
};

//--------------------------------------------------------------------

class WtGraph
{
  public:

    // Constructor
    WtGraph ( int maxNumber = defMaxGraphSize )
        throw ( bad_alloc );

    // Destructor
    ~WtGraph ();

    // Graph manipulation operations
    void insertVertex ( Vertex newVertex )            // Insert vertex
        throw ( logic_error );
    void insertEdge ( char *v1, char *v2, int wt )    // Insert edge
        throw ( logic_error );
    bool retrieveVertex ( char *v, Vertex &vData ) const;
                                                      // Get vertex
    bool edgeWeight ( char *v1, char *v2, int &wt ) const
        throw ( logic_error );                        // Get edge wt.
    bool getEdgeWeight ( char *v1, char *v2, int &wt ) const
        throw ( logic_error );                        // Get edge wt.
    void removeVertex ( char *v )                     // Remove vertex
        throw ( logic_error );
    void removeEdge ( char *v1, char *v2 )            // Remove edge
        throw ( logic_error );
    void clear ();                                    // Clear graph

    // Graph status operations
    bool isEmpty () const;                        // Graph is empty
    bool isFull () const;                         // Graph is full
    bool hasProperColoring () const;              // Proper coloring

    // Output the graph structure -- used in testing/debugging
    void showStructure () const;

  private:

    // Facilitator functions
    int index ( char *v ) const;                // Converts vertex label to an
                                                //   adjacency matrix index
    int getEdge ( int row, int col ) const;     // Set/get edge weight using
                                                //   adjacency matrix indices
    void setEdge ( int row, int col, int wt );  // Set/get edge weight using
                                                //   adjacency matrix indices

    // Data members
    int maxSize,          // Maximum number of vertices in the graph
        size;             // Actual number of vertices in the graph
    Vertex *vertexList;   // Vertex list
    int *adjMatrix;       // Adjacency matrix
};
