//--------------------------------------------------------------------
//
//  Laboratory 8                                         test8.cpp
//
//  Test program for the operations in the Expression Tree ADT
//
//--------------------------------------------------------------------

#include <iostream>
#include <stdexcept>

using namespace std;

//#include "ExprTree.cpp"
#include "ExpressionTree.cpp"
#include "config.h"

//--------------------------------------------------------------------
//  Function prototype

template <typename DataType>
void dummy ( ExprTree<DataType> copyTree );   // copyTree is passed by value

//--------------------------------------------------------------------

int main()
{
#if !LAB8_TEST1 || LAB8_TEST2 || LAB8_TEST3
    // Don't do this if testing boolean tree, unless also testing programming 
    // exercises 2 or 3 (for which this section is mostly needed).
    // The tricky part occurs if testing exercise 1 and (2 or 3), or if
    // someone is trying to test the basic class and one of the other exercises
    // in parallel. Hence the #if expression above.
    cout << "Start of testing the basic expression tree" << endl;
    ExprTree<float> testExpression;  // Test expression

    cout << endl << "Enter an expression in prefix form : ";

    testExpression.build();
    testExpression.showStructure();
    testExpression.expression();
    cout << " = " << testExpression.evaluate() << endl;

     // Test the copy constructor.
     dummy(testExpression);
     cout << endl << "Original tree:" << endl;
     testExpression.showStructure();
#endif

#if LAB8_TEST1
    cout << "Start of testing the boolean expression tree" << endl;
    ExprTree<bool> boolTree;
    cout << endl << "Enter a boolean expression in prefix form : ";
    boolTree.build();
    boolTree.showStructure();
    boolTree.expression();
    cout << " = " << boolTree.evaluate() << endl;
    cout << "** End of testing the boolean expression tree" << endl;
#endif

#if LAB8_TEST2
    cout << "Start of testing commute()" << endl;
     testExpression.commute();
     cout << endl << "Fully commuted tree: " << endl;
     testExpression.showStructure();
     testExpression.expression();
     cout << " = " << testExpression.evaluate() << endl;
    cout << "End of testing commute()" << endl;
#endif

#if LAB8_TEST3
    cout << "Start of testing isEquivalent()" << endl;
    ExprTree<float> same = testExpression;
    cout << "same is equal (tests copy constructor) ?  ";
    cout << (same.isEquivalent(testExpression) ? "Yes" : "No") << endl;

    ExprTree<float> empty;
    cout << "empty is equal?  ";
    cout << (empty.isEquivalent(testExpression) ? "Yes" : "No") << endl;

    ExprTree<float> userExpression;
    cout << "Enter another expression in prefix form: ";
    userExpression.build();
    cout << "new expression is equal?  ";
    cout << (userExpression.isEquivalent(testExpression) ? "Yes" : "No") << endl;
    cout << "** End of testing isEquivalent()" << endl;
#endif

#if !LAB8_TEST1 && !LAB8_TEST2 && !LAB8_TEST3
    // Don't bother with this if testing any of the programming exercises
    cout << endl << "Clear the tree" << endl;
    testExpression.clear();
    testExpression.showStructure();
    cout << "** End of testing the basic expression tree" << endl;
#endif

    return 0;
}

//--------------------------------------------------------------------

template <typename DataType>
void dummy ( ExprTree<DataType> copyTree )

// Dummy routine that is passed an expression tree using call by
// value. Outputs copyTree and clears it.

{
    cout << endl << "Copy of tree:  " << endl;
    copyTree.showStructure();
    copyTree.clear();
    cout << "Copy cleared:   " << endl;
    copyTree.showStructure();
}

