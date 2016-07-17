/*=============================================================================
 Creator:      Terence Henriod
 Course:       CPE301
 Section:      1101
 Program Name: HW9 UART and Cstring
 Description:  
 Revision #:   v0.01
 Date:         10/29/2013
 
 ==============================================================================*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Global Variables (Yes, these are global!)
 Mostly used for register pointers that can't be declared constant.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

unsigned char* RBR = (unsigned char*) 0xFCC0;  // data Recieve BuffeR
                                               // TODO: is this a valid address?

unsigned char* THR = (unsigned char*) 0xFCC0;  // Transmit Holding Register 
                                               // TODO: is this a valid address?

unsigned char* LSR = (unsigned char*) 0xFCC5;  // the UART status register 
                                               // TODO: is this a valid address?


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Constants
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Function Prototypes
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void myUSARTinit( unsigned long ubaud );

unsigned char myKbhit();

unsigned char myGetChar();

void myPutChar( unsigned char vname );

void myPrintString( unsigned char* str0 );

int myStrLen( unsigned char* str0 );

void myStrCpy( int destLen, unsigned char* source, unsigned char* destination );

int myStrCmp( unsigned char* first, unsigned char* second );


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Setup
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void setup()
{
  // initialize the USART0 port
  U0init( 9600 );

  // no return - void
}



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Loop
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void loop()
{
  // variables
  volatile unsigned char cs1; // to hold the data from the terminal
  
  // wait for a kb_hit
  while (U0kbhit()==0){};	// wait for RDA = true

  // once the kb_hit has been detected, read the character value
  cs1 = U0getchar();		// read character

  // put the character in the transmission to the terminal
  //U0putchar(cs1);			// echo character

  // echo the hex value of the char
  echoCharHexValue( cs1 );

  // end of loop function - restart from main
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Function Implementations
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void myUSARTinit( unsigned long ubaud )
{

}


unsigned char myKbhit()
{  
  // return 1 if kbhit detected
  return ( LSR & 0x01 );  // data ready flag in bit 0
}


unsigned char myGetChar()
{
  // return the char data sitting in the RBR
  return *RBR;
}


void myPutChar( unsigned char vname )
{
  // wait for previous data to be transmitted (wait for status flag)
  while( (LSR & 0x40) == 0 )    // TBE flag in bit 6
  {};
  
  // once THR is known to be empty, write the data to the THR to be output
  *THR = vname;
  
  // no return - void
}


void myPrintString( unsigned char* str0 )
{
  // variables
  volatile int ndx = 0;
  
  // find the length
  while( str0[ndx] != '\0' )
  {
    // output the character
    putChar( str0[ndx] );
    
    // increment the index
    ndx++;
  }
  
  // no return - void
}


int myStrLen( unsigned char* str0 )
{
  // variables
  volatile int ndx = 0;
  
  // find the length
  while( str0[ndx] != '\0' )
  {
    // count this character towards the length
    ndx++;
  }
  
  // return the length
  return length;
}


void myStrCpy( int destLen, unsigned char* source, unsigned char* destination )
{
  // variables
  volatile int ndx = 0;
  
  // copy the characters over one at a time
  // this operation may store a truncated version of source
  while( ( source[ndx] != '\0' ) && ( ndx < (destLen - 1) ) )
  {
    destination[ndx] = sorce[ndx];
  }
  
  // if necessary, pad with spaces
  while( ndx < (destLen - 1) )
  {
    // store spaces
    destination[ndx] = ' ';
  }
  
  // terminate the destination string with '\0'
  destination[destLen - 1] = '\0';
  
  // no return - void
}


int myStrCmp( unsigned char* first, unsigned char* second )
{
  // variables
  volatile int result = 0;
  volatile int ndx = 0;
  
  // find the first unequal character
  while( ( first[ndx] == second[ndx] ) &&
         ( ( first[ndx] != '\0' ) &&
           ( second[ndx] != '\0' ) ) )
  {
    // increment the index
    ndx++
  }
  
  // test the first differing character
  // case: the first string is greater than the second
  if( first[ndx] > second[ndx] )
  {
    // result is 1
    result = 1;
  }
  // case: the first string is less than the second
  else if( first[ndx] > second[ndx] )
  {
    // result is -1
    result = -1;
  }
  // case: the strings are the same
    // loop broke at NULL terminators, if first two tests fail, both
    // are equivalent; result is already zero
  
  // return the result
  return result; 
}


