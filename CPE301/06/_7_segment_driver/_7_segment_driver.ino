/*==============================================================================
Creator:      Terence Henriod
Course:       CPE301
Section:      1101
Program Name: 7 Segment Driver
Revision #:   v0.01
Date:         10/9/2013
==============================================================================*/

/*==============================================================================
   GLOBAL CONSTANTS
==============================================================================*/

// bit masks
  // note: many (if not all) of these will be neglected because it is easier
  // to use hex literals rather than define 2^8 constants

// time values (milliseconds)
const unsigned long int HALF_SEC = 500;
const unsigned long int DEBOUNCE = 20;

/*==============================================================================
   FUNCTION PROTOTYPES
==============================================================================*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function's intended use is to accept a byte (represented as an int) that 
will represent hex values 0-F, and the switch statement will configure output
port K such that it will appropriately light a 7-segment display to display 
the given hex digit.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void drive7Segment( volatile unsigned int input );

void myDelay( volatile unsigned long int mSecondsAprox );

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function's intended use is to be called once a change in a port signal is
detected. The function then waits for 20 msec before re-polling, and if the
differing inout is still present, the previous value is updated and returned
by reference and value.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
unsigned int myDebounce( volatile unsigned int* inPort, 
                         volatile unsigned int& previousVal );


/*==============================================================================
   SETUP
==============================================================================*/
void setup()
{
  // variables
  volatile unsigned char* portDDRB;
    portDDRB = (unsigned char*) 0x24;
  volatile unsigned char* portDDRK;
    portDDRK = (unsigned char*) 0x107;
  volatile unsigned char* portK;
    portK = (unsigned char*) 0x108;
    
  // set DDRs
    // using portB bits 0-3 as input (0-3 low)
    *portDDRB &= 0xF0;

    // using portK bits 0-7 as output (0-7 high)
    *portDDRK |= 0xFF;

  // set output (portK 0-7) all high to start
  *portK |= 0xFF;  
}


/*==============================================================================
   MAIN LOOP
==============================================================================*/
void loop()
{
  // variables
  volatile unsigned int* inPortB;
    inPortB = (unsigned int*) 0x23;  // this is the address for the portB input
    *inPortB &= 0xFE;
  static volatile unsigned char charValue = 0x04;
  static volatile unsigned int previous;  

while( true )
{
  // loop through the hex digits to display functionality
  for( charValue = (unsigned char) 0x04, previous = *inPortB;
      charValue <= (unsigned char) 0x0F; charValue++ )
  {
    // display the hex digit
    drive7Segment( charValue );

    // let digit display for a while

    
/*============ The debouncing part ======================================*/
    // check for push button input
    if( *inPortB != previous )
    {
      // delay to let the bouncing taper
      myDelay( DEBOUNCE );
    
      // check again for the changed input
      if( *inPortB != previous )
      {
      // update  the *now* previous value
      previous = *inPortB;
      charValue = (unsigned char) 0x00;
      break;      // will break ALL loops (even for loop), revise
      }
/*============ End the debouncing part ==================================*/   
    myDelay( HALF_SEC);    
    
    }
  }
  
  // if digit F is reached, restart
  if( charValue >= (unsigned char)0x0F )
  {
    charValue = (unsigned char) 0x00;
  }
  
  // use the 7 segment driver
  drive7Segment( charValue );

  // no return - void - continue loop operation
}
}


/*==============================================================================
   FUNCTION IMPLEMENTATIONS
==============================================================================*/

void drive7Segment( volatile unsigned char input )
{
  // variables
  volatile unsigned int* portK;
    portK = (unsigned int*) 0x108;
    
  // given a character, display different behavior
  switch( input )
  {
    case 0x00:
      // display a 0
      
      // set segment a high
      // set segment b high
      // set segment c high
      // set segment d high
      // set segment e high
      // set segment f high
      // (0011 1111)
      *portK |= 0x3F;
      
      // set segment g low
      // (1011 1111)
      *portK &= 0xBF;
      break;
      
    case 0x01:
      // display a 1
      
      // set segment b high
      // set segment c high
      // (0000 0110)
      *portK |= 0x06;

      // set segment a low
      // set segment d low
      // set segment e low
      // set segment f low     
      // set segment g low
      // (1000 0110)
      *portK &= 0x86;
      break;
      
    case 0x02:
      // display a 2
      
      // set segment a high
      // set segment b high
      // set segment d high
      // set segment e high
      // set segment g high
      // (0101 1011)
      *portK |= 0x5B;
      
      // set segment c low
      // set segment f low
      // (1101 1011)
      *portK &= 0xDB;
      break;
      
    case 0x03:
      // display a 3
      
      // set segment a high
      // set segment b high
      // set segment c high
      // set segment d high
      // set segment g high
      // (0100 1111)
      *portK |= 0x4F;

      // set segment e low
      // set segment f low
      // (1100 1111)
      *portK &= 0xCF;
      break;
      
    case 0x04:
      // display a 4

      // set segment b high
      // set segment c high
      // set segment f high
      // set segment g high
      // (0110 0110)
      *portK |= 0x66;
      
      // set segment a low  
      // set segment d low
      // set segment e low
      // (1110 0110)
      *portK &= 0xE6;
      break;
      
    case 0x05:
      // display a 5
      
      // set segment a high
      // set segment c high
      // set segment d high
      // set segment f high
      // set segment g high
      // (0110 1101)
      *portK |= 0x6D;

     // set segment b low
     // set segment e low
     // (1110 1101)
      *portK &= 0xED;
      break;
      
    case 0x06:
      // display a 6
      // set segment a high
      // set segment c high
      // set segment d high
      // set segment e high
      // set segment f high
      // set segment g high
      // (0111 1101)
      *portK |= 0x7D;

      // set segment b low
      // (1111 1101)
      *portK &= 0xFD;      break;
      
    case 0x07:
      // display a 7
      
      // set segment a high
      // set segment b high
      // set segment c high
      // (0000 0111)
      *portK |= 0x07;

      // set segment d low
      // set segment e low
      // set segment f low
      // set segment g low
      // (1000 0111)
      *portK &= 0x87;
      break;
      
    case 0x08:
      // display an 8
      
      // set segment a high
      // set segment b high
      // set segment c high
      // set segment d high
      // set segment e high
      // set segment f high
      // set segment g high
      // (0111 1111)
      *portK |= 0x7F;
      break;
      
    case 0x09:
      // display a 9
      // set segment a high
      // set segment b high
      // set segment c high
      // set segment f high
      // set segment g high
      // (0110 0111)
      *portK |= 0x67;
      
      // set segment d low
      // set segment e low
      // (1110 0111)
      *portK &= 0xE7;
      break;
      
    case 0x0A:
      // display an A
      // set segment a high
      // set segment b high
      // set segment c high
      // set segment e high
      // set segment f high
      // set segment g high
      // (0111 0111)
      *portK |= 0x77;
      
      // set segment d low
      // (1111 0111)
      *portK &= 0xF7;
      break;
      
    case 0x0B:
      // display a b
      
      // set segment c high
      // set segment d high
      // set segment e high
      // set segment f high
      // set segment g high
      // (0111 1100)
      *portK |= 0x7C;
      
      // set segment a low
      // set segment b low
      // (1111 1100)
      *portK &= 0xFC;
      break;
      
    case 0x0C:
      // display a C
      // set segment a high
      // set segment d high
      // set segment e high
      // set segment f high
      // (0011 1001)
      *portK |= 0x39;

      // set segment b low
      // set segment c low
      // set segment g low
      // (1011 1001)
      *portK &= 0xB9;
      break;
      
    case 0x0D:
      // display a d
 
      // set segment b high
      // set segment c high
      // set segment d high
      // set segment e high
      // set segment g high
      // (0101 1110)
      *portK |= 0x5E;

      // set segment a low
      // set segment f low
      // (1101 1110)
      *portK &= 0xDE;
      break;
      
    case 0x0E:
      // display an E
      
      // set segment a high
      // set segment d high
      // set segment e high
      // set segment f high
      // set segment g high
      // (0111 1001)
      *portK |= 0x79;

      // set segment b low
      // set segment c low
      // (1111 1001)
      *portK &= 0xF9;
      break;
      
    case 0x0F:
      // display an F
      
      // set segment a high
      // set segment e high
      // set segment f high
      // set segment g high
      // (0111 0001)
      *portK |= 0x71;

      // set segment b low
      // set segment c low
      // set segment d low
      // (1111 0001)
      *portK &= 0xF1;
      break;
  }
  
  // no return - void
}


void myDelay( volatile unsigned long int mSecondsAprox )
{
  // create a dummy variable, used to delay the program execution via
  // repeated looping
  volatile unsigned long ticker;

  // get a value for the amount of time to wait  
  unsigned long endTime = 1000 * mSecondsAprox;
  
  // delay the program for the specified time by simply reassigning i 
  // many times
  for( ticker = 0; ticker < endTime; ticker ++ );

  // no return - void
}

unsigned int myDebounce( volatile unsigned int* inPort, 
                         volatile unsigned int& previousVal )
{
  // delay to let any bounce settle
  myDelay( DEBOUNCE );

  // check to see if the signal is still different from before
  if( *inPort != previousVal )
  {
    // if it is, accept the value, update the *now* previous value
    previousVal = *inPort;
  }
  
  // return the new previous value for use
  return previousVal;
}
