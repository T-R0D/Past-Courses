/*=============================================================================
 Creator:      Dwight D. Egbert
 Modified by:  Terence Henriod (Annotations)
 Course:       CPE301
 Section:      1101
 Program Name: Echo
 Description:  Echoes serial input characters from hyperterminal keyboard, back
               to hyperterminal display
 Revision #:   v0.01
 Date:         10/22/2013
 
 ==============================================================================*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Global Variables (Yes, these are global!)
 Mostly used for register pointers that can't be declared constant.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

volatile unsigned char *myUCSR0A = (unsigned char *) 0xC0; // Controls/Status
volatile unsigned char *myUCSR0B = (unsigned char *) 0xC1; // Controls/Status
volatile unsigned char *myUCSR0C = (unsigned char *) 0xC2; // Controls/Status
volatile unsigned char *myUBRR0L = (unsigned char *) 0xC4; // Baud Rate
volatile unsigned char *myUBRR0H = (unsigned char *) 0xC5; // Baud Rate
volatile unsigned char *myUDR0   = (unsigned char *) 0xC6; // Data Register


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Constants
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#define RDA 0x80  // Receive Data Available 
#define TBE 0x20  // Transmitter Buffer Empty


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Function Prototypes
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void U0init( unsigned long ubaud );

unsigned char U0kbhit();

unsigned char U0getchar();

void U0putchar( unsigned char U0pdata );

void echoCharHexValue( unsigned input );

unsigned char getFirstHex( unsigned char originalValue );

unsigned char getSecondHex( unsigned char originalValue );


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

void U0init( unsigned long ubaud )
{
  // variables
  unsigned long FCPU = 16000000;  // assumed frequency of the processor in Hz TODO: global constant
  unsigned int tbaud;
  
  // calculate the value to store in the USART Baud Rate Register (this must
  // follow the calculation to be a compatible value)
  tbaud = (((unsigned long)(FCPU / ((unsigned long) (16 * ubaud)))) - 1);
  
  // clear the USART Controls and Status Register A
  *myUCSR0A = 0x20;  // Set the Data Register Empty and Frame Error bits.
                  // clear all other flags: Receive Complete, Transmit Complete,
                  // Data OverRun, Parity Error, Double TX speed, and Multi-
                  // Processor Communication Mode
                   
  // setup USART Controls and Status Register B 
  *myUCSR0B = 0x18;  // only set the Receiver Enable and Transmitter Enable
                  // clear the Interrupt Enables (significant bits) and
                  // set the character size to 5 bits (less significant) TODO: set to 8 bits?

  // setup USART Controls and Status Register C
  *myUCSR0C = 0x06;   // set an 8 bit character size (2:1)
                   // while setting the UMSEL (7:6) to asynchronus mode, disabling
                   // the Parity Mode (5:4), set the Stop Bit Select to 0 (1-bit),
                   // and set the UCPOL (Clock POLarity) to zero to receive sampled
                   // data on the falling edge of the clock and change the
                   // data transmitted on the rising edge
                  
  // load the Baud Rate Registers
  // because arithmetic was done in decimal, we use bitwise operations to keep the desired
  // parts of the number without doing messy decimal to hex kinds of conversions
  *myUBRR0H = (tbaud >> 8); // shifts the calculated value right, leaving the most significant byte
  *myUBRR0L = (tbaud & 0xFF);  // keeps only the 1 bits in low byte of the calculated value
}


unsigned char U0kbhit()
{
  // variables
  unsigned char U0stat; // state of the data register (buffer) 
  
  // ascertain if the Read Data Available 
  U0stat = (*myUCSR0A & RDA);
  
  // return whether or not a 
  return U0stat;
}


unsigned char U0getchar()
{
  // variables
  unsigned char U0gdata; // will contain the read in character
  
  // get the character data from the data register (buffer)
  U0gdata = *myUDR0;
  
  // return the read character
  return U0gdata;
}


void U0putchar( unsigned char U0pdata )
{
  // wait for all data to be read in from the data register (buffer)
  while ((*myUCSR0A & TBE) == 0){};
  
  // once buffer is clear, load the character to be transmitted into the buffer
  *myUDR0 = U0pdata;
}


void echoCharHexValue( unsigned input )
{
  // variables
  unsigned char firstHex;
  unsigned char secondHex;
  
  // break the char into its hex values
  firstHex = getFirstHex( input );
  secondHex = getSecondHex( input );
  
  // output the hex values
  U0putchar( firstHex );
  U0putchar( secondHex );

  // output a space
  U0putchar( (unsigned char) 0x20 );
U0putchar( '!' );

  // no return - void
}


unsigned char getFirstHex( unsigned char originalValue )
{
  // variables
  unsigned char interim;
  unsigned char firstHex = '$';
 
  // get the more significant nibble
  interim = (originalValue & 0xF0);
  
  // shift the nibble down to get the hex number
  interim = (interim >> 4);

  // add 30 hex to convert to the ASCII value of the number
  interim += 0x30;
  
  // store the value
  firstHex = interim;

  // return the char
  return firstHex;
}


unsigned char getSecondHex( unsigned char originalValue )
{
  // variables
  unsigned char interim;
  unsigned char secondHex = '$';

  // AND away the upper hex digit
  interim = (originalValue & 0x0F);
  
  // add hex 30 to get the ASCII value of the digit
  interim += 0x30;

  // save this value
  secondHex = interim;
  
  // return the char
  return secondHex;  
}
