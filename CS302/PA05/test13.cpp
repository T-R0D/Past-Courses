//--------------------------------------------------------------------
//
//  Laboratory 13                                           test13.cpp
//
//  Test program for the operations in the Timer ADT
//
//--------------------------------------------------------------------

#include <iostream>
#include <cctype>
#include <ctime>

using namespace std;

#include "Timer.h"

// Uncomment the following line if running in MS-Windows
// #define WIN32


// wait() is cross platform and works well enough for measuring
// wall clock time, but it is not efficient.
// Feel free to replace it with a routine that works better in
// your environment.
void wait(int secs)
{
  int start = clock();
  while (clock() - start < CLOCKS_PER_SEC * secs);
}

void print_help();

int main()
{

  char cmd;                 // Input command
  bool threwError = false;
  char continueInput;       // Used to pause until user enters data to continue
  int pauseSeconds = 0;     // Wall clock seconds to pause while testing timer

  threwError = false;
  Timer stdTimer;
  Timer unstartedTimer;
  Timer unstoppedTimer;



  print_help();

  do
  {


    cout << endl << "Command ('H' for help): ";
    cin >> cmd;		            // Read command
    cmd = toupper(cmd);           // Upcase so don't have to deal with both cases
    cin.ignore(100, '\n');        // Clear out rest of input buffer

    switch ( cmd )
    {
      case 'H' :
        print_help();
        break;

      case 'P' :
	    // Incorrect way to measure wall clock.
	    // Pause without computing anything. Wall clock will have changed, 
	    // but no process time. Result is zero elapsed process time.
	
	    // Get output out of the way before starting timer
	    cout << "Press <ENTER> to stop timer" << endl;
	    stdTimer.start();
	    cin.get(continueInput);
	    if( continueInput != '\n' ) cin.ignore(100, '\n');
	    stdTimer.stop();
	    cout << "Elapsed process time = " << stdTimer.getElapsedTime() 
	         << " seconds" << endl;
        break;

      case 'R' :
// !! This needs to be completed. !!
	    // Correct way to measure wall clock.
	    // Pause without computing anything. Wall clock will have changed, 
	    // no process time. Result is elapsed wall clock time.
	
	    // Get output out of the way before starting timer
	    cout << "Press <ENTER> to stop timer" << endl;
	    stdTimer.start();
	    cin.get(continueInput);

	    if( continueInput != '\n' )
        {
          cin.ignore(100, '\n');
        }

        stdTimer.stop();
	      cout << "Elapsed process time = " << stdTimer.getElapsedTime() 
	           << " seconds" << endl;

        break;

      case 'W' :
        // Get output out of the way before starting timer
        cout << "Enter number of seconds to pause: ";
        cin >> pauseSeconds;
        stdTimer.start();
        // Pause while doing tight loop. Wall clock will have changed, 
        // but no process time. Result is zero elapsed process time.
        wait( pauseSeconds );
        stdTimer.stop();
        cout << "Elapsed wall clock time = " << stdTimer.getElapsedTime() 
             << " seconds" << endl;
        break;

      case '+' :
        try 
        {
          unstartedTimer.stop();
        }
        catch (logic_error &e)
        {
          threwError = true;        // Record that threw a logic error
          cout << "Exception: message = \"" << e.what() << "\"" << endl;
	      }

        if ( threwError )
        {
	        cout << "stop() correctly threw a logic error." << endl;
	      }
        else
        {
	        cout << "stop() did not throw logic error." << endl;
	      }

        break;

      case '-' :
        try
        {
          (void) unstartedTimer.getElapsedTime();
        }
        catch (logic_error &e)
        {
	        threwError = true;      // Record that threw a logic error
	        cout << "Exception: message = \"" << e.what() << "\"" << endl;
	      }

        if ( threwError )
        {
	        cout << "getElapsedTime() correctly threw a logic error." << endl;
	      }
        else
        {
	        cout << "getElapsedTime() did not throw logic error." << endl;
	      }

        break;

      case '!' :
        try
        {
	      unstoppedTimer.start();
	      }
        catch ( ... )
        {
	        cout << "Error testing timer. Cannot start timer correctly." << endl
	             << "Please fix timer start first. Skipping rest of test." << endl;

          // Skip rest of test.
          continue;
	      }

        try
        {
          (void) unstoppedTimer.getElapsedTime();
        }
        catch (logic_error &e)
        {
		      threwError = true;      // Record that threw a logic error
		      cout << "Exception: message = \"" << e.what() << "\"" << endl;
	      }

        if ( threwError )
        {
          cout << "getElapsedTime() correctly threw a logic error." << endl;
	      }
        else
        {
	        cout << "getElapsedTime() did not throw logic error." << endl;
	      }

        break;

      case 'Q' :
	    // Do nothing. Just don't want to match default
        break;


      default :                               // Invalid command
        cout << "Inactive or invalid command" << endl;
    }

  } while ( cmd != 'Q' );
 
    return 0;
}

void print_help()
{
    cout << endl << "Commands:" << endl;
    cout << "  P   : Test the timer with wall clock time (incorrect)" << endl;
    cout << "  R   : Retest the timer with wall clock time (correct)" << endl;
    cout << "  W   : Test the timer with process clock time" << endl;
    cout << "  +   : Tests stopping timer without having started the timer" << endl;
    cout << "  -   : Tests getting elapsed time without starting the timer" << endl;
    cout << "  !   : Tests getting elapsed time without stopping the timer" << endl;
    cout << "  H   : Print this help screen" << endl;
    cout << "  Q   : Quit the test program" << endl;
    cout << endl;
}

