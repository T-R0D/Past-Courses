//--------------------------------------------------------------------
//
//  Laboratory 12                                         test12.cpp
//
//  Test program for the operations in the Weighted Graph ADT
//
//--------------------------------------------------------------------

#include <iostream>
#include <cstring>
#include <cctype>

using namespace std;

#include "WeightedGraph.h"
#include "config.h"

void print_help();

int main()
{
    WeightedGraph testGraph(8);         // Test graph
    WeightedGraph::Vertex testVertex;            // Vertex
    string v1,
         v2;   // Vertex labels
    char cmd;                     // Input command
    int wt;                       // Edge weight

    print_help();

    do
    {

#if LAB12_TEST1
        testGraph.showShortestPaths();                 // In-lab Exercise 1
#endif

#if LAB12_TEST2
                                                  // In-lab Exercise 2
        if ( testGraph.hasProperColoring() )
           cout << endl << "Proper coloring" << endl;
        else
           cout << endl << "NOT a proper coloring" << endl;
#endif

#if LAB12_TEST2
                                                  // In-lab Exercise 3
        if ( testGraph.areAllEven() )
           cout << endl << "All vertices have even degree" << endl;
        else
           cout << endl << "NOT all vertices have even degree" << endl;
#endif

        testGraph.showStructure();                   // Output graph

        cout << endl << "Command (H for help): ";                 // Read command
        cin >> cmd;
        cmd = toupper( cmd );			     // Normalize to upper case
        if ( cmd == '+'  ||  cmd == '?'  ||  cmd == '-' )
           cin >> v1;
        else if ( cmd == '#'  ||  cmd == '!' )
           cin >> v1 >> v2;
        else if ( cmd == '=' )
           cin >> v1 >> v2 >> wt;

#if LAB12_TEST2
        if ( cmd == '+' ) {                       // In-lab Exercise 2
	   char color;
           cin >> color;
	   testVertex.setColor(color);
	}
#endif

        switch ( cmd )
        {
          case 'H' :
               print_help();
               break;

          case '+' :                                 // insertVertex
               cout << "Insert vertex : " << v1 << endl;
               testVertex.setLabel(v1);
               testGraph.insertVertex(testVertex);
               break;

          case '=' :                                 // insertEdge
               cout << "Insert edge : " << v1 << " " << v2 << " "
                    << wt << endl;
               testGraph.insertEdge(v1,v2,wt);
               break;

          case '?' :                                 // retrieveVertex
               if ( testGraph.retrieveVertex(v1,testVertex) )
                  cout << "Vertex " << v1 << " exists" << endl;
               else
                  cout << "Vertex NOT found" << endl;
               break;

          case '#' :                                 // edgeWeight
               if ( testGraph.getEdgeWeight(v1,v2,wt) )
                  cout << "Weight = " << wt << endl;
               else
                  cout << "No edge between these vertices" << endl;
               break;

          case '-' :                                 // removeVertex
               cout << "Remove vertex " << v1 << endl;
               testGraph.removeVertex(v1);
               break;

          case '!' :                                 // removeEdge
               cout << "Remove the edge between vertices "
                    << v1 << " and " << v2 << endl;
               testGraph.removeEdge(v1,v2);
               break;

          case 'C' :                                 // clear
               cout << "Clear the graph" << endl;
               testGraph.clear();
               break;

          case 'E' :                                 // isEmpty
               if ( testGraph.isEmpty() )
                  cout << "Graph is empty" << endl;
               else
                  cout << "Graph is NOT empty" << endl;
               break;

          case 'F' :                                 // isFull
               if ( testGraph.isFull() )
                  cout << "Graph is full" << endl;
               else
                  cout << "Graph is NOT full" << endl;
               break;

          case 'Q' :                              // Quit test program
               break;

          default :                               // Invalid command
               cout << "Invalid command" << endl;
        }
    }
    while ( cmd != 'Q' );

    return 0;
}

void print_help()
{
#if !LAB12_TESTX
#endif
    cout << endl << "Commands:" << endl;
    cout << "  H       : Help (displays this message)" << endl;
    cout << "  +v      : Insert (or update) vertex v" << endl;
    cout << "  =v w wt : Insert an edge with weight wt between "
         << "vertices v and w" << endl;
    cout << "  ?v      : Retrieve vertex" << endl;
    cout << "  #v w    : Display the weight of the edge between "
         << "vertices v and w" << endl;
    cout << "  -v      : Remove vertex v" << endl;
    cout << "  !v w    : Remove the edge between vertices v and w"
         << endl;
    cout << "  C       : Clear the graph" << endl;
    cout << "  E       : Empty graph?" << endl;
    cout << "  F       : Full graph?" << endl;
    cout << "  Q       : Quit the test program" << endl;
    cout << endl;

}
