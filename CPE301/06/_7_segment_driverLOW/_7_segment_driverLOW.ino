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

// port pointers (unfortunately these are global for ease)  // TODO: make local?
unsigned char* PORT_DDRB  = (unsigned char *) 0x24;
unsigned char* PORT_B = (unsigned char *) 0x25;

unsigned int* PORT_DDRK  = (unsigned int *) 0x107;          // TODO: different type?
unsigned int* PORT_K = (unsigned int *) 0x108;

// time values (milliseconds)
const unsigned long int HALF_SEC = 500;
const unsigned long int DEBOUNCE = 20;

/*==============================================================================
   FUNCTION PROTOTYPES
==============================================================================*/

void drive7Segment( unsigned int input );

void myDelay( unsigned long int milliWait );

unsigned int myDebounce( unsigned int* inPort, unsigned int previousVal );


/*==============================================================================
   SETUP
==============================================================================*/
void setup()
{
  // set DDRs
    // using portB bits 0-3 as input (0-3 low)
    *PORT_DDRB &= 0xF0;

    // using PORT_K bits 0-7 as output (0-7 low)
    *PORT_DDRK |= 0x7F;

  // set output (PORT_K 0-7) all low
  *PORT_K |= 0x7F;  
}


/*==============================================================================
   MAIN LOOP
==============================================================================*/
void loop()
{
  // variables
  unsigned char input = '0';
  unsigned char prevInput = '0';
  
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  This code is for testing purposes.
  Remove to execute actual
  functionality.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  int charValue = int(0x30);
  charValue ++;
  input = (unsigned char) charValue;
  myDelay( HALF_SEC );
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

  prevInput = myDebounce( (unsigned int *) input, prevInput );

  // use the 7 segment driver
  drive7Segment( prevInput );

  // no return - void - continue loop operation
}


/*==============================================================================
   FUNCTION IMPLEMENTATIONS
==============================================================================*/

void drive7Segment( unsigned char input )
{
  // variables
    // none
    
  // given a character, display different behavior
  switch( input )
  {
  case 0x00:
    // display 0
    
    // set all pins high
    *PORT_K |= 0xFF
    
    // set g low
    // (1011 1111)
    *PORT_K &= 0xBF;
    break;

  case 0x01:
    // display 1
    
    // set all pins high
    *PORT_K |= 0xFF
    
    // set b, c low
    // (1111 1001)
    *PORT_K &= 0xF;
    break;
    
  case 0x02:
    // display 2
    
    // set all pins high
    *PORT_K |= 0xFF
    
    // set a, b, d, e, g low
    // (1010 0100)
    *PORT_K &= 0xA4;
    break;
    
  case 0x03:
    // display 3
    
    // set all pins high
    *PORT_K |= 0xFF
    
    // set a, b, c, d, g low
    // (1011 0000)
    *PORT_K &= 0xB0;
    break;

  case 0x04:
    // display 4
    
    // set all pins high
    *PORT_K |= 0xFF
    
    // set  low
    // (1110 0110)
    *PORT_K &= 0xE6;
    break;
    
  case 0x05:
    // display 5
    
    // set all pins high
    *PORT_K |= 0xFF
    
    // set b, e low
    // (1110 1101)
    *PORT_K &= 0xED;
    break;
    
  case 0x06:
    // display 6
    
    // set all pins high
    *PORT_K |= 0xFF
    
    // set b, e low
    // (1110 1101)
    *PORT_K &= 0xED;
    break;
  }
  
  // no return - void
}


void myDelay( unsigned long int milliWait )
{
  // variables
  unsigned long startTime = millis();

  // wait for the specified time period
  while( millis() <= (startTime + milliWait) );
 
  // no return - void
}

unsigned int myDebounce( unsigned int* inPort, unsigned int previousVal )
{
  // check to see if a different signal is present
  if( *inPort != previousVal )
  {
    // delay to let any bounce settle
    myDelay( DEBOUNCE );

    // check to see if the signal is still different from before
    if( *inPort != previousVal )
    {
      // if it is, accept the value
      previousVal = *inPort;
    }
  }
 
  // return the *now* previous value
  return previousVal; 
}
