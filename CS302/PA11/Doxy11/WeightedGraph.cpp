/**
    @file WeightedGraph.cpp

    @author Terence Henriod

    Lab 11: Weighted Graph ADT

    @brief Class implementations for the Weighted Graph ADT.

    @version Original Code 1.00 (11/14/2013) - T. Henriod
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   HEADER FILES / NAMESPACES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// Class Declaration
#include "WeightedGraph.h"
#include "config.h"

// Other Dependencies
#define NDEBUG
#include <cassert>


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================
                   CLASS FUNCTION IMPLEMENTATIONS
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   CONSTRUCTOR(S) / DESTRUCTOR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
WeightedGraph

The default constructor for the WeightedGraph ADT. Initializes an empty graph
of the given parameterized size.

@param maxNumber   The number of vertices the WeightedGraph can contain when
                   full. By default the value MAX_GRAPH_SIZE (defined in
                   WeightedGraph.h)is given.

@pre
-# There is available memory to instantiate a WeightedGraph object.
-# The WeightedGraph is given a valid identifier.

@post
-# An empty WeightedGraph object will be instantiated.

@code
@endcode
*/
WeightedGraph::WeightedGraph( int maxNumber )
{
  // initialize data members
  maxSize = maxNumber;
  size = 0;
  vertexList = new Vertex [ maxSize ];
  adjacencyMatrix = new int [ maxSize * maxSize ];
  pathMatrix = new int [ maxSize * maxSize ];

  // no return - constructor
}


/**
WeightedGraph

The default constructor for the WeightedGraph ADT. Initializes an empty graph
of the given parameterized size.
@param maxNumber   The number of vertices the WeightedGraph can contain when
                   full. By default the value MAX_GRAPH_SIZE (defined in
                   WeightedGraph.h)is given.

@pre
-# There is available memory to instantiate a WeightedGraph object.
-# The WeightedGraph is given a valid identifier.
-# Weightedgraph other is a valid instantiation of a WeightedGraph object.

@post
-# A WeightedGraph object equivalent to the given parameter WeightedGraph other
   will be instantiated.

@code
@endcode
*/
WeightedGraph::WeightedGraph( const WeightedGraph& other )
{
  // initialize data members
  maxSize = 0;
  size = 0;
  vertexList = NULL;
  adjacencyMatrix = NULL;
  pathMatrix = NULL;

  // clone other into this
  *this = other;

  // no return - constructor
}


/**
operator=

The overloaded assignment operator. Clones the data in the given parameter
WeightedGraph other into *this.

@param other   A WeightedGraph to whose data will be cloned.

@return *this   A reference to this for multiple assignments on the same line.

@pre
-# Both *this and other are valid WeightedGraph objects.

@post
-# *this will be an equivalent object to other. All data in *this will be a
   clone of that in other.

@detail @bAlgorithm
-# If the given parameter other is *this, no action is taken.
-# If WeightedGraph other is of different size, *this is resized.
-# The data in this is then made equivalent to that of other, size and all.

@code
@endcode
*/
WeightedGraph& WeightedGraph::operator=( const WeightedGraph& other )
{
  // variables
  int ndx = 0;
  int matrixSize = ( size * size );

  // case: *this is not equivalent to other
  if( this != &other )
  {
    // case: *this has a different max size than other
    if( maxSize != other.maxSize )
    {
      // update the maxSize member
      maxSize = other.maxSize;

      // resize the vertex list array
      delete [] vertexList;
        vertexList = NULL;
      vertexList = new Vertex[ maxSize ];

      // resize the adjacency matrix
      matrixSize = ( other.size * other.size );
      delete [] adjacencyMatrix;
        adjacencyMatrix = NULL;
      adjacencyMatrix = new int[ matrixSize ];

      // resize the path matrix
      delete [] pathMatrix;
        pathMatrix = NULL;
      pathMatrix = new int [ matrixSize ];
    }

    // iterate across the vertex list of other
    for( size = 0; size < other.size; size++ )
    {
      // copy the current vertex
      vertexList[ size ] = other.vertexList[ size ];
    }

    // iterate across the adjacency matrix
    for( ndx = 0; ndx < matrixSize; ndx++ )
    {
      // copy the elements of the adjacency matrix
      adjacencyMatrix[ ndx ] = other.adjacencyMatrix[ ndx ];
    }

    // iterate across the path matrix
    for( ndx = 0; ndx < matrixSize; ndx++ )
    {
      // copy the elements of the adjacency matrix
      pathMatrix[ ndx ] = other.pathMatrix[ ndx ];
    }
  }
  // otherwise, take no action

  // return *this
  return *this;
}


/**
WeightedGraph

The destructor for the WeightedGraph ADT. Ensures all dynamic memory is
returned.

@pre
-# *this was a properly instantiated WeightedGraph object.

@post
-# All dynamic memory will be returned.
-# *this will be destructed.

@code
@endcode
*/
WeightedGraph::~WeightedGraph ()
{
  // return vertexList memory
  delete [] vertexList;
    vertexList = NULL;

  // return adjacency matrix memory
  delete [] adjacencyMatrix;
    adjacencyMatrix = NULL;

  // return the path matrix' dynamic memory
  delete [] pathMatrix;
    pathMatrix = NULL;


  // no return - destructor
}



/*
WeightedGraph::Vertex::Vertex( const string& newLabel, const char newColor  )
{
  // initialize the data members
  label = newLabel;
  color = newColor;

  // no return - constructor
}


WeightedGraph::Vertex::Vertex( const Vertex& other )
{
  // call the assignment operator
  *this = other;
}

WeightedGraph::Vertex& WeightedGraph::Vertex::operator=( const Vertex& other )
{
  // case: *this is not being copied to itself
  if( this != &other )
  {
    // update the members of *this
    label = other.label;
    color = other.color;
  }

  // return *this
  return *this;
}
*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   MUTATORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
insertVertex

Adds a vertex to the graph if there is room in the list. An exception is thrown
if parameter Vertex newVertex cannot be added. No edges are added, just a
vertex.

@param newVertex   A new vertex to be added to the graph.

@pre
-# A valid WeightedGraph object has been instantiated.
-# The WeightedGraph has not been filled to maximum capacity.

@post
-# If possible, Vertex newVertex is added to the end of the vertexList array
   member.
-# If the graph is full, an exception of type logic_error is thrown to indicate
   that the graph is full.

@detail @bAlgorithm
-# If the vertexList array member is not full, then the size of the list is
   increased by one and the vertex is added in the last spot in the list.
-# A row of "non-edges" is then added to the adjacencyMatrix array member.
-# An edge with the weight of a loop cost is added where the new vertex
   intersects with itself in the matrix.
-# A column of "non_edges" is also appended to the matrix.
-# If the vertexList was full, then an exception of type logic_error is thrown
   to indicate that the WeightedGraph is full.

@exception logic_error   Indicates that an attempt to add a Vertex to a full
                         WeightedGraph was made

@code
@endcode
*/
void WeightedGraph::insertVertex( const Vertex& newVertex ) throw( logic_error )
{
  // variables
  int newNdx = NOT_FOUND;
  int ndx = 0;

  // search for the vertex in the existing list
  newNdx = getIndex( newVertex.getLabel() );

  // case: the vertex was found in the list
  if( newNdx != NOT_FOUND )
  {
    // update the date with the given data
    vertexList[ newNdx ] = newVertex;

    // clear any edges that were previously associated with the vertex
    for( ndx = 0; ndx < size; ndx++ )
    {
      // case: the current index is the same as the new vertex' index
      if( ndx == newNdx )
      {
        // set a loop-weight edge
        setEdge( ndx, newNdx, INFINITE_EDGE_WT );
      }
      // case: the current vertex and the new vertex are different
      else
      {
        // set a "non-edge" between the new vertex and the current vertex
        setEdge( newNdx, ndx, INFINITE_EDGE_WT );
        setEdge( ndx, newNdx, INFINITE_EDGE_WT );
      }
    }
  }
  // case: the vertex is not in the list and the list is not full
  else if( size < maxSize )
  {
    // track the new vertex's new index
    newNdx = size;

    // add the new vertex
    vertexList[ newNdx ] = newVertex;

    // increment the size of the graph
    size++;

    // append a new row to the matrix
    for( ndx = 0; ndx < newNdx; ndx++ )
    {
      // add a row entry indicating no edge
      setEdge( newNdx, ndx, INFINITE_EDGE_WT );
    }
    setEdge( newNdx, ndx, LOOP_COST ); // don't forget the "loop"

    // append a new column
    for( ndx = 0; ndx < newNdx; ndx++ )
    {
      // add a column entry indicating no edge
      setEdge( ndx, newNdx, INFINITE_EDGE_WT );
    }

/* GOES WITH A BAD IMPLEMENTATION METHOD, BUT IT WAS HARD SO I'M PROUD OF
   THIS CODE
    // append the row of no edges (and the "loop") to the matrix that
    // corresponds to the new edge
    for( columnNdx = 0; columnNdx < ( size - 1 ); columnNdx++ )
    {
      // indicate that there is no edge
      setEdge( newNdx, columnNdx, INFINITE_EDGE_WT );
    }
    setEdge( newNdx, columnNdx, 0 );

    // iterate through the rows of the new matrix, copying back data as
    // as appropriate
    for( ndx = ( ( ( newNdx - 1 ) * size ) + newNdx ), rowNdx = ( newNdx - 1 );
         rowNdx >= 0;
         rowNdx-- )
    {
      // iterate through all the columns
      for( columnNdx = newNdx; columnNdx >= 0; columnNdx--, ndx-- )
      {
        // case: the matrix location represents an edge between the new vertex
        //       and an older vertex
        if( columnNdx == newNdx )
        {
          // store a "non-edge"
          setEdge( rowNdx, newNdx, INFINITE_EDGE_WT );
        }
        // otherwise, copy back the data appropriately
        else
        {
          // copy the item back
          adjacencyMatrix[ ndx ] = adjacencyMatrix[ ndx - rowNdx ];
        }
      }
    }
*/

  }
  // case: a vertex was not updated and the vertex list is full
  else
  {
    // throw exception to indicate that the graph is full
    throw logic_error( "Unable to add vertex to full graph." );
  }

  // no return - void
}


/**
insertEdge

Inserts an undirected edge of the given weight between the given vertices.

@param v1   The label of the first vertex in the pair defining the edge.
@param v2   The label of the second vertex in the pair defining the edge.
@param wt   The weight of the edge to be updated or added.

@pre
-# *this is a properly instantiated WeightedGraph.
-# The graph contains at least two vertices.
-# The vertices that define the edge should be present in the graph.

@post
-# The weighted edge will be added to the graph. That is, the adjacency matrix
   will contain the updated/new edge weight in the two matrix locations
   pertaining to the given vertices (because this is not a digraph).
-# If either of the given vertex parameters are not found, then an exception of
   type logic_error is thrown to indicate such.

@detail @bAlgorithm
-# The vertex list is searched for both vertices and their index is recorded.
-# The found indices are used to locate the appropriate locations in the
   adjacency matrix.
-# The adjacency matrix is then updated with the appropriate edge weight.
-# If either of the vertices is not found in the graph, then an exception of
   type logic_error is thrown to indicate that the edge cannot be added between
   vertices that do not exist in the graph.

@exception logic_error   This exception is used to report that the edge cannot
                         be added between vertices that do not exist in the
                         graph.

@code
@endcode
*/
void WeightedGraph::insertEdge( const string& v1, const string& v2,
                                const int wt ) throw( logic_error )
{
  // variables
  int v1Ndx = NOT_FOUND;
  int v2Ndx = NOT_FOUND;
  bool edgeExists = false;

  // search for vertex 1
  v1Ndx = getIndex( v1 );

  // case: vertex 1 was found
  if( v1Ndx != NOT_FOUND )
  {
    // search for vertex 2
    v2Ndx = getIndex( v2 );

    // case: both vertices were found
    if( v2Ndx != NOT_FOUND )
    {
      // then the edge exists in the graph
      edgeExists = true;

      // set both of the edge weights in the equivalent locations (since this is
      // not a digraph, the matrix is symmetric)
      setEdge( v1Ndx, v2Ndx, wt );
      setEdge( v2Ndx, v1Ndx, wt );
    }

  // output some made up debugging garbage that shouldn't be because...
  cout << "Size = " << size << ", idx_v1 = " << getIndex( v1 )
       <<  ", idx_v2 = " << getIndex( v2 ) << endl;
  }

  // case: at least one vertex was not found
  if( !edgeExists )
  {
    // throw an exception to indicate that the edge was not added
    throw( "Unable to add edge due to missing vertices." );
  }

  // no return - void
}


/**
retriveVertex

Searches the vertex list for a particular vertex label. Passes the vertex back
by reference if it is found.

@param v       A string used to identify a sought vertex.
@param vData   An object of type Vertex intended to contain the data of the
               sought vertex if the vertex with a label matching the parameter
               label is found.

@return result   The truth value of whether the sought vertex was found. If the
                 vertex was found and vData contains that vertex's information,
                 then true is returned. Otherwise, false is returned and vData
                 is in an undetermined state.

@pre
-# *this is a valid instantiation of a weighted graph.
-# vData is a properly constructed Vertex object.
-# A vertex with a label matching the one given as a parameter should exist in
   the graph.

@post
-# If a vertex with a matching label is found, then its data is copied into the
   given reference parameter Vertex vData and true is returned. Otherwise, false
   is returned and vData will be in an undetermined state.

@detail @bAlgorithm
-# The vertex list is searched for a vertex with a label matching the given
   parameter.
-# If such a vertex is found, its data is copied to parameter Vertex vData and 
   true is returned.
-# If a vertex with such a label is not found, then parameter Vertex vData will
   be in an undetermined state and false is returned.

@code
@endcode
*/
bool WeightedGraph::retrieveVertex( const string& v, Vertex& vData ) const
{
  // variables
  int ndx = NOT_FOUND;
  bool result = false;

  // search the vertex list for a vertex with a matching label
  ndx = getIndex( v );

  // case: a match was found
  if( ndx != NOT_FOUND )
  {
    // indicate that a matching vertex was found
    result = true;

    // copy the data into the reference parameter to be passed back
    vData = vertexList[ ndx ];
  }

  // return the result
  return result;
}


/**
getEdgeWeight

Finds the edge wieght corresponding to the edge between two vertices of the
given labels. The weight is then passes back by reference. true is retruned to
indicate that the edge (or at least the two vertices that define it) exist in
the graph. Otherwise, false is returned.

@param v1   A label for the first vertex used to define an edge.
@param v2   A label for the second vertex used to define an edge.
@param wt   The parameter used to pass back a weight of a sought edge.

@return result   The truth value of whether an edge could be found in a graph,
                 true if the edge does exist in the graph, false otherwise. Also
                 indicates if the reference parameter int wt contains valid or
                 undeterminate data.

@pre
-# *this is a valid instantiation of a WeightedGraph.
-# The sought vertices v1 and v2 (and therefore the edge between them) should
   exist in the graph.

@post
-# If the edge exists in the graph, then its weight is copied into the int wt
   reference parameter to be passed back and true is returned.
-# Otherwise, wt will be in an undetermined state and false is returned.

@detail @bAlgorithm
-# The vertex list is searched for verticex with lables matching the two given
   ones.
-# If found, their indices are used to locate the edge wait in the adjacency
   matrix.
-# The edge weight is copied to the reference parameter wt.
-# If either of the vertices is not found, then an exception is thrown to
   indicate that an invalid edge was sought.

@exception logic_error   Used to indicate that at least one of the given vertex
                         labels does not exist in the graph, and therefore the
                         edge does not exist.

@code
@endcode
*/
bool WeightedGraph::getEdgeWeight( const string& v1, const string& v2, int& wt )
                                   const throw( logic_error )
{
  // variables
  int v1Ndx = NOT_FOUND;
  int v2Ndx = NOT_FOUND;
  bool result = false;

  // search for vertex 1
  v1Ndx = getIndex( v1 );

  // case: vertex 1 was found
  if( v1Ndx != NOT_FOUND )
  {
    // search for vertex 2
    v2Ndx = getIndex( v2 );

    // case: both vertices were found
    if( v2Ndx != NOT_FOUND )
    {
      // pass the edges weight back by reference
      wt = getEdge( v1Ndx, v2Ndx );

      // case: the edge's weight intdicates a valid edge
      if( wt != INFINITE_EDGE_WT )
      {
        // the search for an edge was successful
        result = true;
      }
      // case: the edge's weight indicates a non-edge
        // an edge was not successfully found, do nothing (see vriable
        // initialization)
    }
  }
  // case: at least one vertex was not found
  if( ( v1Ndx == NOT_FOUND ) || ( v2Ndx == NOT_FOUND ) )
  {
    // throw an exception to indicate that the edge was not added
    throw( "Unable to add edge due to missing vertices." );
  }

  // return result
  return result;
}


/**
removeVertex

Removes a vertex from the vertex list and resizes the adjacency matrix to remove
any edges that were associated with the vertex.

@param v   A label corresponding to the vertex to be removed.

@pre
-# *this is a valid instantiation of a WeightedGraph.
-# The given label parameter should correspond to a vertex contained in the
   vertex list of the graph.

@post
-# The vertex corresponding to the given label will be removed.
-# Data corresponding to edges shared by the removed vertex and other vertices
   will be removed and the adjacency matrix will be resized appropriately.
-# The new size of the graph is updated.
-# If the given label does not correspond to a given vertex, an exception is
   thrown to indicate this.

@detail @bAlgorithm
-# The sought vertex is first located in the vertex list and its index is saved.
-# The size of the graph is decremented.
-# The vertex list is then condensed by shifting all elements in the list after
   the removed vertex over.
-# The rows of the adjacency matrix that appear after the row corresponding to
   the removed vertex are moved up to replace the "removed" row.
-# The columns that appear after the one corresponding to the removed vertex are
   shifted left in order to replace the "removed" column.
-# If the given label is not found in the graph, an exception is thrown,
   indicating this.

@exception logic_error   Used to indicate that an attemp to remove a vertex that
                         is not present in the graph was made.

@code
@endcode
*/
void WeightedGraph::removeVertex( const string& v ) throw( logic_error )
{
  // variables
  int rowNdx = 0;
  int columnNdx = 0;
  int vNdx = NOT_FOUND;

  // search for the vertex corresponding to the given label
  vNdx = getIndex( v );

  // case: v was found
  if( vNdx != NOT_FOUND )
  {
    // decrement the size of the graph
    size--;

    // condense the vertex list
    for( rowNdx = vNdx; rowNdx < size; rowNdx++ )
    {
      vertexList[ rowNdx ] = vertexList[ rowNdx + 1 ];
    }

    // shift the matrix rows up as appropriate
    for( rowNdx = vNdx; rowNdx < size; rowNdx++ )
    {
      // shift up each entry
      for( columnNdx = 0; columnNdx <= size; columnNdx++ )
      {
        // copy the entry up
        setEdge( rowNdx, columnNdx, getEdge( ( rowNdx + 1 ), columnNdx ) ); 
      }
    }

    // shift columns (row by row) left as appropriate
    for( rowNdx = 0; rowNdx < size; rowNdx++ )
    {
      // shift each entry left as appropriate to eliminate a column
      for( columnNdx = vNdx; columnNdx < size; columnNdx++ )
      {
        // shift the entry
        setEdge( rowNdx, columnNdx, getEdge( rowNdx, ( columnNdx + 1 ) ) );
      }
    }

/* GOES WITH A BAD IMPLEMENTATION METHOD, BUT IT WAS HARD SO I'M PROUD OF
   THIS CODE
    // iterate through all rows of the matrix to perform resizing
    for( ndx = 0, offset = 0, rowNdx = 0; rowNdx < size; rowNdx++ )
    {
      // case: the row about to be process corresponds to the vertex being
      //       removed
      if( rowNdx == vNdx )
      {
        // update the offset value to skip the row of data to be removed
        offset += ( size + 1 );
      }

      // iterate through the columns of the matrix
      for( columnNdx = 0; columnNdx < size; columnNdx++, ndx++ )
      {
        // case: the current matrix location corresponds to the vertex being
        //       removed
        if( columnNdx == vNdx )
        {
          // update the offset value to accomodate for this
          offset++;
        }

        // move back the data appropriately
        adjacencyMatrix[ ndx ] = adjacencyMatrix[ ndx + offset ];
      }
    }
*/

  }
  // case: v was not found
  else
  {
    // throw an exception to indicate that an invalid removal attempt was made
    throw logic_error( "Unable to remove a vertex that is not in the graph." );
  }

  // no return - void
}


/**
removeEdge

Removes an edge from the graph by giving it an "infinite" weight. (The
"infinite" weight is the maximum integer value defined in climits)

@param v1   A label corresponding to the first vertex that defines an edge.
@param v2   A label corresponding to the second vertex that defines an edge.

@pre
-# *this is a valid instantiation of a WeightedGraph.
-# Vertices with labels corresponding to the parameters v1 and v2 should exist
   in the graph.

@post
-# If the edge exists in the graph, it is "removed" by giving it an "infinite"
   weight.
-# If either of the given vertex labels are not present in the vertex list, an
   exception is thrown to indicate that an invalid edge was sought.

@detail @bAlgorithm
-# The insertEdge function is called to "remove" the edge by giving it an
   infinite weight. This will prevent duplication of code.
-# If the either of the vertices that define the edge can't be found, then an
   exception is thrown to indicate such.

@exception logic_error   Used to indicate that an invalid edge was chosen
                         because at least one of the given vertex labels does
                         not correspond to an existing vertex.

@code
@endcode
*/
void WeightedGraph::removeEdge( const string& v1, const string& v2 )
    throw( logic_error )
{
  // variables
  int v1Ndx = NOT_FOUND;
  int v2Ndx = NOT_FOUND;
  bool edgeExists = false;

  // search for vertex 1
  v1Ndx = getIndex( v1 );

  // case: vertex 1 was found
  if( v1Ndx != NOT_FOUND )
  {
    // search for vertex 2
    v2Ndx = getIndex( v2 );

    // case: both vertices were found
    if( v2Ndx != NOT_FOUND )
    {
      // then the edge exists in the graph
      edgeExists = true;

      // set both of the edge weights in the equivalent locations (since this is
      // not a digraph, the matrix is symmetric)
      setEdge( v1Ndx, v2Ndx, INFINITE_EDGE_WT );
      setEdge( v2Ndx, v1Ndx, INFINITE_EDGE_WT );
    }
  }

  // case: at least one vertex was not found
  if( !edgeExists )
  {
    // throw an exception to indicate that the edge was not added
    throw( "Unable to add edge due to missing vertices." );
  }

  // no return - void
}


/**
clear

Empties the WeightedGraph of data.

@pre
-# A valid WeightedGraph object has been instantiated.
 
@post
-# The WeightedGraph will be emptied. 

@detail @bAlgorithm
-# The size of the data set is reduced to zero.

@code
@endcode
*/
void WeightedGraph::clear()
{
  // "empty" the graph
  size = 0;

  // no return - void
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                   ACCESSORS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/**
isEmtpy

Returns the truth of the emptiness of the graph, true if empty, false otherwise.

@return empty   The truth of the graph being empty.

@pre
-# *this is a valid instantiation of a WeightedGraph.
-# The size member of *this is in a valid state.

@post
-# The truth of the emptiness of the graph is returned.

@code
@endcode
*/
bool WeightedGraph::isEmpty() const
{
  // return the truth of the graph being empty
  return ( size == 0 );
}


/**
isFull

Returns the truth of the fullness of the graph, true if full, false otherwise.

@return full   The truth of the graph being full.

@pre
-# *this is a valid instantiation of a WeightedGraph.
-# The size member of *this is in a valid state.

@post
-# The truth of the fullness of the graph is returned.

@code
@endcode
*/
bool WeightedGraph::isFull() const
{
  // return the truth of the fullness of the graph
  return ( size == maxSize );
}


/**
showStructure

Outputs a graph's vertex list and adjacency matrix. This operation is intended
for testing/debugging purposes only.

PROVIDED BY THE LAB MANUAL PACKAGE

@pre
-# *this is a valid instantiation of a WeightedGraph.
 
@post
-# *this will remain unchanged.
-# The vertex list an adjacency matrix for the graph will be displayed.
-# The weighted graph will remain unchanged.

@detail @bAlgorithm
-# A vertex list is displayed, both labels and colors
-# Then the adjacency matrix is displayed one row at a time.
-# If the graph is actually empty, this is stated.

@code
@endcode
*/
void WeightedGraph::showStructure() const
{
  // case: the graph is empty
  if ( size == 0 )
  {
    cout << "Empty graph" << endl;
  }
  // case: the graph is not empty
  else
  {
    // display the vertex list
    cout << endl << "Vertex list : " << endl;
    for ( int row = 0 ; row < size ; row++ )
    {
      cout << row << '\t' << vertexList[row].getLabel()
#if LAB12_TEST2 
           << "\t" << vertexList[row].getColor()
#endif
           << endl;
    }

    // display the edge matrix header
    cout << endl << "Edge matrix : " << endl << '\t';
    for ( int col = 0 ; col < size ; col++ )
    {
      cout << col << '\t';
    }
    cout << endl;

    // display the edge matrix row by row
    for ( int row = 0 ; row < size ; row++ )
    {
      // display row label
      cout << row << '\t';

      // display each element of the row
      for ( int col = 0 ; col < size ; col++ )
      {
        // get the weight of the row
        int wt = getEdge(row,col);

        // case: the element weight represents a non-edge.
        if ( wt == INFINITE_EDGE_WT || wt == 0 )
        {
          // display a non-existant edge
          cout << "- \t";
        }
        // case: the edge has a valid weight
        else
        {
          // display that weight
          cout << wt << '\t';
        }
      }

      // move down for next row
      cout << endl;
    }
  }

  // no return - void
}


/**
computePaths

Computes the shortest paths between the vertices of a graph and stores them in
the graph's path matrix.

@pre
-# *this is a valid instantiation of a WeightedGraph
-# The adjacency matrix is in a valid state.
 
@post
-# The shortest (lowest cost) paths between all vertices will be computed.
-# The path costs will be stored in the path matrix.
-# No record of what the shortest paths actually are will be kept, only the
   resulting cost.

@detail @bAlgorithm
Floyd's Algorithm is used as follows:
-# Every vertex is iteratively considered as a "mutual" neighbor.
-# Every vertex is iteratively considered for every given "mutual" vertex as a
   "start" vertex.
-# Every other vertex is then iteratively considered for given "mutual" and
   "start" vertices, as an "end" vertex.
-# A check is made for the existence of paths between the start vertex and the
   mutual vertex and between the mutual vertex and the end vertex.
-# If such paths exist, then the cost of traversing the two paths described in
   the previous step (the "indirect" path) is compared with the cost of the
   path directly from the start vertex to the end vertex.
-# If the indirect path is found to be cheaper than the direct one, then the
   direct one is replaced with the indirect one on the path matrix.

@code
@endcode
*/
void WeightedGraph::computePaths()
{
  // variables
  int m = 0;
  int j = 0;
  int k = 0;
  int matrixSize = ( maxSize * maxSize );

  // copy all the elements of the adjacency matrix
  for( m = 0; m < matrixSize; m++ )
  {
    // copy the current item
    pathMatrix[ m ] = adjacencyMatrix[ m ];
  }

  // ***implement Floyd's Algorithm***
  // check all possibilities for a given possible mutual neighbor vertex m
  for( m = 0; m < size; m++ )
  {
    // check all possible chains for a given vertex j
    for( j = 0; j < size; j++ )
    {
      // check all neighbors of j, some vertex k
      for( k = 0; k < size; k++ )
      {
        // case: there is a between the given vertex j and the mutual neighbor m
        if( getPath( j, m ) != INFINITE_EDGE_WT )
        {
          // case: there is a chain between the mutual neighbor m and some
          //       vertex k
          if( getPath( m, k ) != INFINITE_EDGE_WT )
          {
            // case: the chain from the given vertex j to the mutual neighbor m
            //       to some vertex k is cheaper than the chain from the given
            //       vertex j directly to some vertex k
            if( (unsigned long long)( getPath( j, m ) + getPath( m, k ) ) <
                getPath( j, k ) )
            {
              // replace the direct chain cost with the indirect one if it is
              // cheaper
              setPath( j, k, ( getPath( j, m ) + getPath( m, k ) ) );
            }
          }
        }
      }
    }
  }
}


/**
showShortestPaths

Computes and displays the graphs path matrix.

@pre
-# *this is a valid instantiation of a WeightedGraph
-# The adjacency matrix is in a valid state.
 
@post
-# The WeightedGraph will remain unchanged.
-# The shortest (lowest cost) paths between all vertices will be re-computed.
-# The shortest path matrix will be displayed.

@detail @bAlgorithm
-# The shortest paths are computed and stored in the pathMatrix data member.
-# Then the path matrix is displayed one row at a time.
-# If the graph is actually empty, this is stated.

@code
@endcode
*/
void WeightedGraph::showShortestPaths()
{
  // re-compute the paths to ensure current data
  computePaths();

  // display the paths
  // case: the graph is empty
//  if ( size == 0 )
//  {
//    cout << "Empty graph" << endl;
//  }
  // case: the graph is not empty
//  else
//  {
    // display the path matrix header
    cout << endl << "Path matrix : " << endl << '\t';
    for ( int col = 0 ; col < size ; col++ )
    {
      cout << col << '\t';
    }
    cout << endl;

    // display the path matrix row by row
    for ( int row = 0 ; row < size ; row++ )
    {
      // display row label
      cout << row << '\t';

      // display each element of the row
      for ( int col = 0 ; col < size ; col++ )
      {
        // get the cost of the path
        int wt = getPath(row,col);

        // case: the path cost represents a non-edge.
        if ( wt == INFINITE_EDGE_WT )
        {
          // display a non-existant path
          cout << "- \t";
        }
        // case: the path has a valid cost
        else
        {
          // display that weight
          cout << wt << '\t';
        }
      }

      // move down for next row
      cout << endl;
    }
//  }

  // no return - void
}


/**
hasProperColoring

Returns the truth value of whether or not the graph has a valid vertex coloring.
true is returned if no vertex is adjacent to another vertex of same color,
returns false otherwise.

@return hasValidColoring   The truth value of whether or not the graph has a
                           valid vertex coloring.


@pre
-# *this is a valid instantiation of a WeightedGraph.
-# The vertices are all colored with valid colors.
 
@post
-# The WeightedGraph will remain unchanged.
-# If the graph has a valid vertex coloring, true is returned. Otherwise, false
   is returned.

@detail @bAlgorithm
-# Every vertex is checked to see if it has the same color as any neighbors.
-# If a pair of neighbors are found to have the same color, the check is halted.
-# If no pair of neighbors are found to have the same color, then true is
   returned.
-# If any pair of neighbors is found to have the same color, false is returned.
TODO refine this algorithm if possible.

@code
@endcode
*/
bool WeightedGraph::hasProperColoring() const
{
  // variables
  int rowNdx = 0;
  int columnNdx = 0;
  bool hasValidColoring = true;

  // iterate across each row of the adjacency matrix
  for( rowNdx = 0; hasValidColoring && ( rowNdx < size ); rowNdx++ )
  {
    // iterate across each column, update matrix index each time
    for( columnNdx = 0;
         hasValidColoring && ( columnNdx < size ) ;
         columnNdx++ )
    {
      // case: the current element of the adjacency matrix is not a diagonal
      if( rowNdx != columnNdx )
      {
        // case: there is a path between the vertex specified by the row index
        //       and the one currently being checked
        if( getEdge( rowNdx, columnNdx ) != INFINITE_EDGE_WT )
        {
          // case: the colors of the vertices match
          if( vertexList[ rowNdx ].getColor() ==
              vertexList[ columnNdx ].getColor() )
          {
            // the graph does not have a valid coloring
            hasValidColoring = false;
          }
        }
      }
    }
  }

  // return the truth of the graph having a valid coloring
  return hasValidColoring;
}


/**
areAllEven

Determines if each vertex in the graph has an even degree, and then returns true
if each vertex does have an even degree and false otherwise.

@return hasEulerianChain   The truth value of whether or not the graph has all
                           even degree vertices (also indicates if there is a
                           closed chain that uses all paths once). true if all
                           vertices do have even degree, false otherwise.

@pre
-# *this is a valid instantiation of a WeightedGraph.
 
@post
-# The WeightedGraph will remain unchanged.
-# The truth of all verticex having even degree is returned.

@detail @bAlgorithm
-# Every vertex is iteratively checked.
-# The number of edges each vertex has is checked.
-# If any vertex is found to have an odd degree the check is halted.
-# Return true if all vertices have even degree, return false otherwise.
TODO refine this algorithm if possible.

@code
@endcode
*/
bool WeightedGraph::areAllEven() const
{
  // variables
  bool hasEularianChain = true;
  int rowNdx = 0;
  int columnNdx = 0;
  int degree = 0;


  // iterate through every vertex (every row)
  for( rowNdx = 0; hasEularianChain && ( rowNdx < size ); rowNdx++ )
  {
    // iterate through every possible neighbor (column)
    for( degree = 0, columnNdx = 0; columnNdx < size; columnNdx++ )
    {
      // case: there is an edge between the vertices
      if( getEdge( rowNdx, columnNdx ) != INFINITE_EDGE_WT )
      {
        // case: the edge isn't a loop
        if( rowNdx != columnNdx )
        {
          // count the degree
          degree++;
        }
      }
    }

    // case: the counted number of edges is not even
    if( ( degree % 2 ) == 1 )
    {
      // indicate that there is no eularian chain
      hasEularianChain = false;
    }
  }

  // return the truth value of the existence of an eularian chain (all vertices
  // have even degree)
  return hasEularianChain;
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            PRIVATE HELPER FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*=====  Clone Helper  =======================================================*/
  // none


/*=====  Mutators  ===========================================================*/

/**
setEdge

Given appropriate adjacency matrix coordinates, sets the weight of the edge
between two vertices.

@param row   The row index of the first vertex defining the edge.
@param col   The column index of the second vertex defining the edge.
@param wt    The weight the sought edge will be given.

@pre
-# *this is a valid instantiation of a WeightedGraph.
-# Parameters row and col are valid indices corresponding to a location in the
   adjacency matrix.

@post
-# The location in the adjacency matrix representing the edge from the row index
   vertex to the column index vertex is given the parameterized weight.

@detail @bAlgorithm
-# The indices that correspond to a 2-D matrix can be converted using the
   formula: (row-index * number-of-columns) + column-index.
-# The resulting index is used to set the edge's weight.

@code
@endcode
*/
void WeightedGraph::setEdge( const int row, const int col, const int wt )
{
  // assert the pre-condition(s) of the function
  assert( ( row >= 0 ) && ( row < size ) );
  assert( ( col >= 0 ) && ( col < size ) );

  // set the desired edge weight with the given parameter
  adjacencyMatrix[ ( ( row * maxSize ) + col ) ] = wt;

  // no return - void
}


/**
setPath


Given appropriate adjacency matrix coordinates, sets the cost of the path
between two vertices.


@param row     The row index of the first vertex defining the path.
@param col     The column index of the second vertex defining the path.

@param cost    The cost the sought path will be given.

@pre
-# *this is a valid instantiation of a WeightedGraph.
-# Parameters row and col are valid indices corresponding to a location in the
   path matrix.

@post
-# The location in the path matrix representing the edge from the row index
   vertex to the column index vertex is given the value of the parameter cost.

@detail @bAlgorithm
-# The indices that correspond to a 2-D matrix can be converted using the
   formula: (row-index * number-of-columns) + column-index.
-# The resulting index is used to set the path's cost.


@code
@endcode
*/
void WeightedGraph::setPath( const int row, const int col, const int cost )
{
  // assert the pre-condition(s) of the function
  assert( ( row >= 0 ) && ( row < size ) );
  assert( ( col >= 0 ) && ( col < size ) );

  // set the desired edge weight with the given parameter
  pathMatrix[ ( ( row * maxSize ) + col ) ] = cost;

  // no return - void
}


/*=====  Accessors  ==========================================================*/

/**
getIndex

Retrieves the index of a vertex in the vertex list for use as a coordinate in an
adjacency matrix.

@param v   A string label used a a key for finding a desired vertex.

@return result   The integer index of the sought vertex in the vertex list.

@pre
-# *this is a valid instantiation of a WeightedGraph.

@post
-# The WeightedGraph will remain unchanged.
-# The index of the sought vertex in the vertex list is returned.
-# If the vertex is not found, NOT_FOUND (defined in WeightedGraph.h) is
   returned to signal that a vertex with a label matching v was not found.

@detail @bAlgorithm
-# A linear search is conducted to find a vertex in the vertex list with a label
   matching the given parameter string v.

@code
@endcode
*/
int WeightedGraph::getIndex( const string& v ) const
{
  // variables
  bool found = false;
  int ndx = 0;
  int result = NOT_FOUND;

  // search for the vertex
  while( !found && ( ndx < size ) )
  {
    // case: current label matches the sought one
    if( v == vertexList[ ndx ].getLabel() )
    {
      // indicate that the vertex was found
      found = true;

      // record its index
      result = ndx;
    }

    // update the index
    ndx++;
  }

  // return the result of the vertex search
  return result;
}


/**
getEdge

Returns the edge weight at the given position at the specified location in the
adjacency matrix.

@param row   The row index to be searched in the path matrix.
@param col   The column index to be searched in the path matrix.

@return weight

@pre
-# *this is a valid instantiation of a WeightedGraph.
-# row and col are valid positions in the adjacency matrix.
 
@post
-# The WeightedGraph and all its members will remain unchanged.
-# The index of the desired entry in the 1-D representation of the adjacency
   matrix is returned.

@detail @bAlgorithm
-# The indices that correspond to a 2-D matrix can be converted using the
   formula: (row-index * number-of-columns) + column-index.

@code
@endcode
*/                                       
int WeightedGraph::getEdge( int row, int col ) const
{
  // assert the pre-condition(s) of the function
  assert( ( row >= 0 ) && ( row < size ) );
  assert( ( col >= 0 ) && ( col < size ) );

  // return the edge weight at the given matrix location
  return ( adjacencyMatrix[ ( ( row * maxSize ) + col ) ] );
}


/**
getPath

Returns the path cost at the given position at the specified location in the
path matrix.

@param row   The row index to be searched in the path matrix.
@param col   The column index to be searched in the path matrix.

@return pathCost

@pre
-# *this is a valid instantiation of a WeightedGraph.
-# row and col are valid positions in the adjacency matrix.
 
@post
-# The WeightedGraph and all its members will remain unchanged.
-# The index of the desired entry in the 1-D representation of the adjacency
   matrix is returned.

@detail @bAlgorithm
-# The indices that correspond to a 2-D matrix can be converted using the
   formula: (row-index * number-of-columns) + column-index.

@code
@endcode
*/                                       
int WeightedGraph::getPath( int row, int col ) const
{
  // assert the pre-condition(s) of the function
  assert( ( row >= 0 ) && ( row < size ) );
  assert( ( col >= 0 ) && ( col < size ) );

  // return the edge weight at the given matrix location
  return ( pathMatrix[ ( ( row * maxSize ) + col ) ] );
}

