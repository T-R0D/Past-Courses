/**
    @file CS677_HW5_code.cpp

    @author Terence Henriod

    CS677 HW5: Dynamic Programming

    @brief This program explores the use of dynamic programming techniques.

    @version Original Code 1.00 (4/10/2014) - T. Henriod
*/

/*==============================================================================
=======     HEADER FILES     ===================================================
==============================================================================*/
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cstdio>

/*==============================================================================
=======     GLOBAL CONSTANTS     ===============================================
==============================================================================*/
const int NO_VALUE = -999;
const int NUMBER_OF_CHOICES = 3;
const int WEEKS_TO_PLAN = 5;
enum JobType
{
  LOW_STRESS = 0,
  HIGH_STRESS,
  REST
};

/*==============================================================================
=======     USER DEFINED TYPES     =============================================
==============================================================================*/



/*==============================================================================
=======     FUNCTION PROTOTYPES     ============================================
==============================================================================*/

/**
Max

Finds the maximum element of an array using a divide and conquer
strategy.

@param A           The array under consideration
@param left_idx    The index of the left-most element considered
                   for a given recursive call
@param right_idx   The index of the right-most element considered
                   for a given recursive call
*/
int findOptimalRevenue( int job_payouts[2][5], int solutions[3][5],
                        int choices[3][5] );


std::pair<int, int> findOptimumChoice( int weekly_revenues[3][5], int week_i );

void printRevenuesTable( int solutions[3][5] );

void printPreviousChoicesTable( int choices[3][5] );

void printJobType( int job_type_code );

void printJobSelections( int choices[3][5], int week_number, int final_choice );




int findOptimalRevenueBEST( int job_payouts[2][5], int number_of_weeks );



/*==============================================================================
=======     MAIN FUNCTION     ==================================================
==============================================================================*/

/**
main

Simply calls the assigned functions with the specified test input given
to give sample output for the homework problems.
*/
int main( int argc, char** argv )
{
  // variables
  int job_payouts[2][5] = {{0, 10, 1, 10, 10}, {0, 5, 50, 5, 1}};
  int solutions[3][5];  // is save results of all 3 choices instead of
  int choices[3][5];    // P[i-1] and P[i-2] so I can reference Rest(P[i-1])

/* I know that this is rather inflexible, but if I had my way, I would do
   it all with vectors or structs that might increase difficulty of grading,
   use stacks instead of recursion, etc. Instead I settled for hacking it
   together to make things fit the assignment problems better.
 */

  // test the solution
  std::cout << "Solution: " << std::endl;
  findOptimalRevenue(job_payouts, solutions, choices );
  std::cout << std::endl << std::endl;

  // end the program
  return 0;
}


/*==============================================================================
=======     FUNCTION IMPLEMENTATIONS     =======================================
==============================================================================*/
int findOptimalRevenue( int job_payouts[2][5], int solutions[3][5],
                        int choices[3][5] )
{
/* NOTE: even though the recursive formula uses the optimal choice from two
         weeks previous, I will just keep a running "rest" choice row in the
         table which will be used to denote the best optimum solution up to the
         two weeks previous.
 */

  // variables
  std::pair<int, int> previous_result;
  std::pair<int, int> low_stress_option;
  std::pair<int, int> high_stress_option;
  int i = 0;

  // setup the "zero" week of choices
  for( i = 0; i <= REST; ++i )
  {
    // set the revenue earned so far to zero and the choice to rest
    solutions[i][0] = 0;
    choices[i][0] = REST;  // part C
  }

  // go through the choices for each week considering optimum solutions to
  // previous weeks
  for( i = 1; i < 5; ++i )
  {
    // find the optimum result of choosing a low stress job
    previous_result = findOptimumChoice( solutions, (i - 1) );
    solutions[LOW_STRESS][i] = previous_result.first +
                               job_payouts[LOW_STRESS][i];
    choices[LOW_STRESS][i] = previous_result.second;  // part C

    // fill in the rest/do nothing portion of the table(s)
    solutions[REST][i] = previous_result.first; // + 0
    choices[REST][i] = previous_result.second;  // part C

    // find the optimum result of choosing a high stress job
    solutions[HIGH_STRESS][i] = solutions[REST][i - 1] +
                               job_payouts[HIGH_STRESS][i];
    choices[HIGH_STRESS][i] = REST;  // part C
  }

  // get the information associated with the optimum revenue for the whole set
  previous_result = findOptimumChoice( solutions, i - 1 );

  // print the resulting optimum choice value
  printf( "(Part B) The highest revenue plan will generate an "
          "income of: %d\r\n", previous_result.first );  // Part B

  // print the results table that displays the values used in finding
  // the optimal solution
  printf( "(Part B) The revenue outcomes for each choice of each week\r\n"
          "(assuming the optimal choice was made in the week prior):\r\n" );
  printRevenuesTable( solutions ); // Part B
  printf( "\r\n" );

  // print the choices prior to each week that led to the optimal solution
  printf( "(Part C) The choices made prior to each week:\r\n" );
  printPreviousChoicesTable( choices );  // Part C
  printf( "\r\n" );

  // print the choices made to reach the solution
  printf( "(Part D) The choices that lead to maximum revenue at week 4:\r\n" );
  printJobSelections( choices, 4, previous_result.second ); // Part D
  printf( "\r\n" );

  // return the optimal plan value
  return previous_result.first;
}


std::pair<int, int> findOptimumChoice( int weekly_revenues[3][5], int week_i )
{
  //variables
  std::pair<int, int> optimum_choice( NO_VALUE, REST );
  int job_choice = 0;

  // initialize the pair
  optimum_choice.first = NO_VALUE;
  optimum_choice.second = REST;

  // check every option to see which one has the highest value
  for( job_choice = 0; job_choice <= REST; ++job_choice )
  {
    // case: we have found a better solution, use it and keep track of
    //       which decision resulted in this one
    if( optimum_choice.first < weekly_revenues[job_choice][week_i] )
    {
      // update the optimum choice
      optimum_choice.first = weekly_revenues[job_choice][week_i];
      optimum_choice.second = job_choice;
    }
  }

  // return the resulting pair
  return optimum_choice;
}


void printRevenuesTable( int solutions[3][5] )
{
  // variables
  int i = 0;

  // print the top of the table
  printf( "%11s  ", "Week" );
  for( i = 1; i < 5; ++i )
  {
    printf( "%2d  ", i ); 
  }
  printf( "\r\n" );

  // print a divider
  for( i = 0; i < 28; ++i )
  {
    printf( "-" );
  }
  printf( "\r\n" );

  // print the low stress row of the table
  printf( "%11s |", "Low-stress" );
  for( i = 1; i < 5; ++i )
  {
    printf( "%2d  ", solutions[LOW_STRESS][i] ); 
  }
  printf( "\r\n" );

  // print the high stress row of the table
  printf( "%11s |", "High-stress" );
  for( i = 1; i < 5; ++i )
  {
    printf( "%2d  ", solutions[HIGH_STRESS][i] ); 
  }
  printf( "\r\n" );

  // print the rest stress row of the table
  printf( "%11s |", "Rest" );
  for( i = 1; i < 5; ++i )
  {
    printf( "%2d  ", solutions[REST][i] ); 
  }
  printf( "\r\n" );

  // no return - void
}


void printPreviousChoicesTable( int choices[3][5] )
{
  // variables
  int i = 0;

  // print an explanation of the table
  printf( "  Values in cells indicate which job was chosen the week "
          "prior to week i.\r\n" );
  printf( "  Row labels to the right indicate the type of job choice "
          "for the week i.\r\n" );

  // print the top of the table
  printf( "%11s  ", "Week" );
  for( i = 1; i < 5; ++i )
  {
    printf( "%11d  ", i ); 
  }
  printf( "\r\n" );

  // print a divider
  for( i = 0; i < 70; ++i )
  {
    printf( "-" );
  }
  printf( "\r\n" );

  // print the low stress row of the table
  printf( "%11s |", "Low-stress" );
  for( i = 1; i < 5; ++i )
  {
    printf( " " ); 
    printJobType( choices[LOW_STRESS][i] );
    printf( " " );
  }
  printf( "\r\n" );

  // print the high stress row of the table
  printf( "%11s |", "High-stress" );
  for( i = 1; i < 5; ++i )
  {
    printf( " " ); 
    printJobType( choices[HIGH_STRESS][i] );
    printf( " " );
  }
  printf( "\r\n" );

  // print the rest stress row of the table
  printf( "%11s |", "Rest" );
  for( i = 1; i < 5; ++i )
  {
    printf( " " ); 
    printJobType( choices[REST][i] );
    printf( " " ); 
  }
  printf( "\r\n" );

  // no return - void
}


void printJobSelections( int choices[3][5], int week_number, int final_choice )
{
  // variables

  //case: we still have not reached the beginning of the list
  if( week_number > 1 )
  {
    // get the selection for the previous week
    printJobSelections( choices, (week_number - 1),
                        choices[final_choice][week_number] );

    // print this week's selection
    printf( " -> Week %d: ", week_number );

    // print the type of job chosen
    printJobType( final_choice );
  }
  // case: we are at the first week
  else if( week_number == 1 )
  {
    // print this week's selection
    printf( "Week %d:", week_number );

    // print the type of job chosen
    printJobType( final_choice );
  }
}


void printJobType( int job_type_code )
{
  // case: the choice was a low-stress job
  if( job_type_code == LOW_STRESS )
  {
    printf( "%11s", "Low-Stress" );
  }
  // case: the choice was a high-stress job
  else if( job_type_code == HIGH_STRESS )
  {
    printf( "%11s", "High-Stress" );
  }
  // case: the choice was to rest
  else
  {
    printf( "%11s", "Rest" );
  }

  // no return - void
}


int findOptimalRevenueBEST( int job_payouts[2][4], int number_of_weeks )
{
  // variables
  int optimum_revenue = 0;
  // note: P[i] is the best revenue for a plan up to week i
  int p_of_i_minus_one = 0;  // since in part a we care only about the
  int p_of_i_minus_two = 0;  // optimal value, I decided to save space
  int low_stress_option = 0;
  int high_stress_option = 0;
  int i = 0;

  // find the revenue values for each choice for the "zero" week
  p_of_i_minus_one = 0;
  p_of_i_minus_two = 0;


  // continue finding the optimal solution for the rest of the weeks
  for( i = 1; i < number_of_weeks; ++i )
  {
    // find the result of choosing a low stress job
    low_stress_option = job_payouts[LOW_STRESS][i] + p_of_i_minus_one;

    // find the result of choosing a high stress job
    high_stress_option = job_payouts[HIGH_STRESS][i] + p_of_i_minus_two;

    // keep the larger of the two
    optimum_revenue = ( high_stress_option > low_stress_option ?
                            high_stress_option : low_stress_option );

    // update the i-1 and i-2 values for the next loop pass
    p_of_i_minus_two = p_of_i_minus_one;
    p_of_i_minus_one = optimum_revenue;
  }

  // report the optimum value found
  return optimum_revenue;
}

