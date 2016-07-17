//--------------------------------------------------------------------
//
//  Laboratory 13                                        wtgraph.cpp
//
//  SOLUTION: Adjacency matrix implementation of the Weighted
//            Graph ADT
//
//--------------------------------------------------------------------

#include <iostream>
#include <cstring>
#include "wtgraph.h"

using namespace std;

//--------------------------------------------------------------------

void WtGraph:: showStructure () const

// Outputs a graph's vertex list and adjacency matrix. This operation
// is intended for testing/debugging purposes only.

{
    int wt,         // Edge weight
        row, col;   // Loop counters

    if ( size == 0 )
       cout << "Empty graph" << endl;
    else
    {
       cout << endl << "Vertex list : " << endl;
       for ( row = 0 ; row < size ; row++ )
           cout << row << '\t' << vertexList[row].label << endl;

       cout << endl << "Edge matrix : " << endl << '\t';
       for ( col = 0 ; col < size ; col++ )
           cout << col << '\t';
       cout << endl;
       for ( row = 0 ; row < size ; row++ )
       {
           cout << row << '\t';
           for ( col = 0 ; col < size ; col++ )
           {
               wt = getEdge(row,col);
               if ( wt == infiniteEdgeWt )
                  cout << "- \t";
               else
                  cout << wt << '\t';
           }
           cout << endl;
       }
    }
}

//--------------------------------------------------------------------
//
//  Facilitator functions
//

int WtGraph:: index ( char *v ) const

// Returns the adjacency matrix index for vertex v. Returns size if
// the vertex does not exist.

{
    int j;  // Loop counter

    for ( j = 0 ;
          j < size  &&  strcmp(vertexList[j].label,v) != 0 ;
          j++ );
    return j;
}

