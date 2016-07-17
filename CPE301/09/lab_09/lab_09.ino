/*=============================================================================
 Creator:      Terence Henriod
 Course:       CPE301
 Section:      1101
 Program Name: HW10 ADC
 Description:  This program demonstrates the use of the Arduino ADC by changing
               the rate at which hex digits are cycled on a 7-segment display.
 Revision #:   v0.01
 Date:         11/5/2013
 
 THIS CODE MAY BE ASSUMED TO BE WRITTEN UNDER THE GNU PUBLIC LISCENCE AND IS
 THEREFORE FREE. FREE DOES NOT NECESSARILY MEAN FREE OF COST, BUT FREE FOR
 DISTRIBUTION AND MODIFICATION. HOWEVER, THE AUTHOR OF THIS CODE MUST BE CITED
 WHEN THIS CODE IS USED, DISTRIBUTED, OR MODIFIED.
 
==============================================================================*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~   Program/Hardware Settings   ~~~~~*/
const unsigned int BAUD_RATE = 9600;
const unsigned long int CPU_FREQ = 16000000;

/*~~~~~   Boolean Values   ~~~~~*/
const unsigned int TRUE = 1;
const unsigned int FALSE = 0;

/*~~~~~   Integer Values   ~~~~~*/
const int MAX_STR = 80;
const int DEFAULT_SETTING = 2;
const int TCNT_MAX = 0xFFFF;                              // MAX counter val
const int DEBOUNCE = 20;                                  // miliseconds
const int NUM_CURSOR_SPEEDS = 5;
const int CURSOR_DELAYS[ NUM_CURSOR_SPEEDS ] =
  { 54000, 48000, 32000, 16000, 8000 };                   // in syswrk() calls
const int NUM_BLINK_DELAYS = 10;
const int BLINK_DELAYS[ NUM_BLINK_DELAYS ] =
  { 1000, 666, 500, 333, 250, 125, 100, 75, 50, 25 };     // in milliseconds
const int NUM_TONE_PERIODS = 7;
const int TONE_PERIODS[ NUM_TONE_PERIODS] =
  { 2272, 2028, 1912, 1704, 1517, 1433, 1277 };           // in microseconds      TODO: fix

/*~~~~~   Character Values   ~~~~~*/
const unsigned char BS = 0x08;
const unsigned char ESC = 0x1B;
const unsigned char cursorPosition[4] = { '|', '/', '-', '\\' };

/*~~~~~   Necessary Strings   ~~~~~*/
const char ENDL[] = "\r\n";
const char SEPARATOR[] =
"\r\n------------------------------------------------------------------\r\n\n";
const char WELCOME[] =
  "Welcome to Terence's Arduino Mega Command Processor\r\n";
const char PLEASE[] = 
  "Please enter the desired function:\r\n\n";
const char SELECT[] =
  "SelectCommand >";
const char BAD_INPUT[] =
  " ...Please select a valid option that is listed...";
const char OPTION1[] =
  "1. Keyboard Echo\r\n";
const char OPTION2[] =
  "2. ASCII Character Echo\r\n";
const char OPTION3[] =
  "3. Change the Rotating Cursor Speed\r\n";
const char OPTION4[] =
  "4. Blink LED\r\n";
const char OPTION5[] =
  "5. Sound Tone\r\n";


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Global Variables (Yes, these are global!)
 Mostly used for register pointers that can't be declared constant.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~   GPIO Registers   ~~~~~*/
volatile unsigned char* portInB  = (unsigned char*) 0x23;
volatile unsigned char* DDR_B    = (unsigned char*) 0x24;
volatile unsigned char* portOutB = (unsigned char*) 0x25;

volatile unsigned char* portInK  = (unsigned char*) 0x106;
volatile unsigned char* DDR_K    = (unsigned char*) 0x107;
volatile unsigned char* portOutK = (unsigned char*) 0x108;

/*~~~~~   Timer/Counter Registers   ~~~~~*/
volatile unsigned char* myTCCR1A = (unsigned char*) 0x80; // status/controlA
volatile unsigned char* myTCCR1B = (unsigned char*) 0x81; // status/controlB
volatile unsigned char* myTIFR1 = (unsigned char*)  0x36; // interrupts/flags
volatile unsigned int*  myTCNT1 = (unsigned int*)   0x84; // for both H and L

/*~~~~~   USART Registers   ~~~~~*/
volatile unsigned char* myUCSR0A = (unsigned char *) 0xC0; // Controls/StatusA
volatile unsigned char* myUCSR0B = (unsigned char *) 0xC1; // Controls/StatusB
volatile unsigned char* myUCSR0C = (unsigned char *) 0xC2; // Controls/StatusC
volatile unsigned char* myUBRR0L = (unsigned char *) 0xC4; // Baud RateL
volatile unsigned char* myUBRR0H = (unsigned char *) 0xC5; // Baud RateH
volatile unsigned char* myUDR0   = (unsigned char *) 0xC6; // Data Register

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Function Prototypes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~   Initialization Functions   ~~~~~*/
void tc1init();
void USART0init( unsigned int uBaud );

/*~~~~~   Essential Functions   ~~~~~*/
unsigned char kb_hit();
unsigned char getch();
void putch( unsigned char data );
void myDelay( unsigned int units, unsigned int useMicro );

/*~~~~~   String Functions   ~~~~~*/
void printString( const char* str0 );
unsigned int strLen( const char* str0 );
unsigned int strCopy( const char* source, char* destination );
void deleteLine( int lineLength );

/*~~~~~   Operational Functions   ~~~~~*/
void printMenu();
void clearScreen();
void syswrk( unsigned int* speedSetting );
void crson();
void crsoff();
void crsrot();
void cmdproc( volatile int* cursorSpeed );

/*~~~~~   Program Functionality Support Functions   ~~~~~*/
unsigned char getFirstHexDigit( unsigned char originalValue );
unsigned char getSecondHex( unsigned char originalValue );
unsigned char settingToChar( int setting );
void genRectangleWavePulse( unsigned int period, unsigned int dutyCycle );

/*~~~~~   Program Functionality   ~~~~~*/
void keyEcho();
void ASCIIecho();
void setCursorSpeed( volatile int* cursorSpeed );
void blinkLED();
void soundTone();

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void setup()
{
  // initialize timer/counter1
  tc1init();
  
  // initialize the USART
  USART0init( BAUD_RATE );

  // initialize GPIO sound output
  *DDR_K    |= 0xF0;  // use upper nibble of K for sound output
  *portOutK &= 0x0F;  // turn sound off to start

  // initialize LED output
  *DDR_B    |= 0x80;   // LED @ pin 7
  *portOutB &= 0x7F;   // turn LED off to start

  // no return - void
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void loop()
{
  // variables
  volatile unsigned char keepChecking = TRUE;
  volatile unsigned int result = 0;
  static int spdSet = DEFAULT_SETTING;
  volatile static int* speedSetting = &spdSet; // instead of ref or global
  
  // clear screen and display the menu
  putch( 0x0C );  // NewPage character
  printMenu();

  // start the cursor
  crson();

  // continually check for a key press
  while( keepChecking )
  {
    // if a key was pressed, take appropriate action    
    if( kb_hit() )
    {
      // stop looking for menu input
      keepChecking = FALSE;
      
      // process the input
      cmdproc( speedSetting );
    }
    // if no key was pressed
    else
    {
      // do system work
      syswrk( *speedSetting );
    }
  }
  
  // end of loop function - restart from main
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Function Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*###########################################   Initialization Functions   ###*/
void tc1init()
{
  // ensure clock is stopped, setup normal mode
  *myTCCR1B = 0; // all bits need to be zero to disable other functionality,
                 // but WGM13 and 11 in bits 4:3 for normal mode
                 // also, writing 000 to CS12:10 in bits 2:1 stops clock
  *myTCCR1A = 0; // all bits need to be zero to disable other functionality,
                 // but WGM11 and 10 in bits 1:0 for normal mode
  // clear the timer overflow flag (write 1) and disable interrupts
  *myTIFR1 = 0x01; // TOV is in bit 0
  
  // no return - void
}

void USART0init( unsigned int uBaud )
{
  // variables
  unsigned int baudRegVal;
  
  // clear the USART Controls and Status Register A
  *myUCSR0A = 0x20; // Set the Data Register Empty and Frame Error bits.
                    // clear all other flags: Receive Complete,
                    // Transmit Complete, Data OverRun, Parity Error,
                    // Double TX speed, and Multi-Processor Communication Mode

  // setup USART Controls and Status Register B 
  *myUCSR0B = 0x18; // only set the Receiver Enable and Transmitter Enable
                    // clear the Interrupt Enables (significant bits) and
                    // set the character size to 5 bits (less significant) TODO: set to 8 bits?

  // setup USART Controls and Status Register C
  *myUCSR0C = 0x06; // set an 8 bit character size (2:1)
                    // while setting the UMSEL (7:6) to asynchronus mode, disabling
                    // the Parity Mode (5:4), set the Stop Bit Select to 0 (1-bit),
                    // and set the UCPOL (Clock POLarity) to zero to receive sampled
                    // data on the falling edge of the clock and change the
                    // data transmitted on the rising edge

  // compute the baud rate register value
    /* Note: using the given formula:
       (((unsigned long)(CPU_FREQ / ((unsigned long)( 16 * uBaud )))) - 1)
       the value to load the baud rate register with can be done, however,
       the computation is prone to error, so it may be easiest to simply
       select the appropriate register value for a given baud rate.
    */
  // case: 2400 baud
  if( uBaud == 2400 )
  {
    // load register appropriately
    baudRegVal = 416;    
  }
  else if( uBaud == 4800 )
  {
    // load register appropriately
    baudRegVal = 207;    
  }
  else if( uBaud == 9600 )
  {
    // load register appropriately
    baudRegVal = 103;    
  }
  else if( uBaud == 115200 )
  {
    // load register appropriately
    baudRegVal = 8;    
  }
  // case: a valid value was not given
  else
  {
    // use 9600 baud by default
    baudRegVal = 103;
  }

  // load the baud rate value into the baud rate registers
  *myUBRR0H = ( baudRegVal >> 8 );   // put only the high byte in high register
                                     // ((1111 0000) >> 8) == (0000 1111)
  *myUBRR0L = ( baudRegVal & 0xFF ); // store only the low byte in low register
  
  // no return - void
}


/*################################################   Essential Functions   ###*/
unsigned char kb_hit()
{
  // return the truth of Read Data Available flag in bit 7 
  return ( *myUCSR0A & 0x80 );
}

unsigned char getch()
{
  // wait for a kb_hit
  while( !kb_hit() ) {};
  
  // return the data read from Rx register
  return *myUDR0;
}

void putch( unsigned char data )
{
  // wait for the transmit buffer to empty (TBE flag in bit 5)
  while( ( *myUCSR0A & 0x20 ) == FALSE ) {};
  
  // write the data to the buffer to output it
  *myUDR0 = data;

  // no return - void
}

void myDelay( unsigned int units, unsigned int useMicro )
{
  // variables
  unsigned int countStart = 0;
  long numTicks = 0;
  unsigned char prescale = 0x05;  // prescale of 1024 for milliseconds
  double ticks = 15.625;          // ticks per time unit
    if( useMicro == TRUE )
    {
      prescale = 0x02; // give a prescale of 8 for microsecond timing
      ticks    = 0.5;    
    }
  
  // ensure timer is stopped and TOV is clear
  *myTCCR1B &= 0xF8; // stop clock only
  *myTIFR1   = 0x01; // clear TOV in bit 0
  
  // calculate number of ticks in desired wait
  /* This means we need to figure out how many ticks are necessary.
     Assuming the frequency of the clock in the Arduino is 16.000 MHz,
     this means that we get 16,000,000 ticks per second, 16 ticks
     per millisecond. 
   
     If we use a pre-scale factor of 1024, then we perceive the timer
     to oscillate at 15625 Hz. Invert this to find that every tick is then
     perceived to occur every 6.4 E-5 seconds.
       
     Now, we need to find a constant that will be correspond to the number
     of ticks per millisecond. This is found by dividing 1 miliisecond
     by the time it takes for a tick. This is 15.625. I will "ceiling" this
     number (if necessary) and say that there are 16 ticks per millisecond. (10h)
    */
  numTicks = (long)( units * ticks );
  
  // compute the value to start the timer/counter at
  countStart = (unsigned int)( (long)(TCNT_MAX + 1) - numTicks );
  
  // store the starting value
  *myTCNT1 = countStart;
  
  // clear TOV, start timer/counter
  *myTIFR1  |= 0x01; // TOV in bit 0
  *myTCCR1B |= prescale;   // use value from above, appy to CS2:0 in 2:0
  
  // wait for the TOV flag to raise
  while( ( *myTIFR1 & 0x01 ) == FALSE ) {};
  
  // stop the timer/counter, clear the TOV flag
  *myTCCR1B &= 0xF8; // stop clock only
  *myTIFR1   = 0x01; // clear TOV in bit 0
  
  // no return - void
}


/*###################################################   String Functions   ###*/
void printString( const char* str0 )
{
  // variables
  volatile unsigned int ndx = 0;

  // output all characters up to NULL
  while( str0[ ndx ] != '\0' )
  {
    // output the character
    putch( str0[ ndx ] );
    
    // move to the next one
    ndx++;
  }
  
  // no return - void
}


unsigned int strLen( const char* str0 )
{
  // variables
  volatile unsigned int ndx = 0;

  // output all characters up to NULL
  while( str0[ ndx ] != '\n' )
  {
    // move to the next one
    ndx++;
  }
  
  // return the number of characters
  return ndx;
}


unsigned int strCopy( const char* source, char* destination )
{
  // variables
  unsigned int success = FALSE;
  volatile unsigned int ndx = 0;

  // if the destination has room for the source, perform the copy
  if( strLen( destination ) >= strLen( source ) )
  {
    // copy up to NULL
    while( source[ ndx ] != '\n' )
    {
      // copy the character
      destination[ ndx ] = source[ ndx ];

      // move to the next character
      ndx++;
    }
   
    // place the NULL terminator
    destination[ ndx ] = '\n';
    
    // operation was successful
    success = TRUE;
  }
  
  // return the success of the operation
  return success;
}

void deleteLine( int lineLength )
{
  // variables
  unsigned int counter = 0;
 
  // output enough backspaces to clear the line
  while( counter <= lineLength )
  {
    // move back
    putch( BS );
    
    // overwrite with a space
    putch( ' ' );
    
    // move back again
    putch( BS );
    
    // increment the count of characters deleted
    counter++;
  } 
}


/*##############################################   Operational Functions   ###*/
void printMenu()
{
  // print the separator
  printString( SEPARATOR );

  // welcome the user
  printString( WELCOME );
  
  // display menu options
  printString( PLEASE );
  printString( OPTION1 );
  printString( OPTION2 );
  printString( OPTION3 );
  printString( OPTION4 );
  printString( OPTION5 );
  printString( ENDL );
  
  // prompt for selection
  printString( SELECT ); 
 
  // no return - void 
}

void clearScreen()
{
  // variables
  unsigned int counter = 0;
 
  // attempt print many items
  while( counter < 50 )
  {
    // print an endline
    printString( ENDL );
  }
 
  // no return - void 
}

void syswrk( unsigned int speedSetting )
{
  // variables
  volatile static unsigned int counter = CURSOR_DELAYS[ speedSetting ];
  
  // decrement the syswrk call counter
  counter--;
  
  // case: it is time to rotate the cursor
  if( counter == 0 )
  {
    // reset the counter to the appropriate delay
    counter = CURSOR_DELAYS[ speedSetting ];
    
    // rotate the cursor
    crsrot(); 
  }
  
  // no return - void
}


void crson()
{
  // delete the prompty arrow ('>')
  putch( BS );
  
  // display the first cursor position
  putch( cursorPosition[ 0 ] );
  
  // no return - void
}


void crsoff()
{
  // delete the cursor character
  putch( BS );
  
  // print the prompt arrow
  putch( '>' );
 
  // no return - void 
}


void crsrot()
{
  // variables
  volatile static unsigned char crsPosition = 0;
  
  // increment the position
  crsPosition++;
  
  // clear the old cursor
  putch( BS );
  
  // print the new one
  putch( cursorPosition[ ( crsPosition & 0x03 ) ] );
  
  // no return - void
}


void cmdproc( volatile int* cursorSpeed )
{
  // variables
  unsigned char response = '$'; // debugging value

  // read the user's response
  response = getch();
 
  // turn cursor off
  crsoff();
  
  // execute the selected command
  if( response == '1' )
  {
    keyEcho();
  }
  else if( response == '2' )
  {
    ASCIIecho();
  }
  else if( response == '3' )
  {
    setCursorSpeed( cursorSpeed );
  }
  else if( response == '4' )
  {
    blinkLED();
  }
  else if( response == '5' )
  {
    soundTone();
  }
  else
  {
    // report that invalid input has been entered
    printString( BAD_INPUT );

    // wait to clear the message when a key press is detected
    getch();
    deleteLine( strLen( BAD_INPUT ) );
  }
  
  // no return - void
}


/*############################   Program Functionality Support Functions   ###*/
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

unsigned char settingToChar( int setting )
{
  // variables
  unsigned char result;
  
  // get the low order digit from the number
  result = ( setting & 0xFF );
  
  // add 31h to convert to a digit starting at 1
  result += 0x31;
  
  // if the result is 10, for now store an '!' to indicate 10 TODO: fix this
  if( result == 0x3A )
  {
    // change result to '!'
    result = '!';  
  }
  
  // return the char
  return result;
}

void genRectangleWavePulse( unsigned int period, unsigned int dutyCycle )
{
  // variables
  volatile unsigned long int hiDelay = 0;
  volatile unsigned long int loDelay = 0;
  volatile unsigned long int counter = 0;

  // scale the high and low periods according to the duty cycle
  hiDelay = (unsigned long int) ((dutyCycle / 100) * period);  
  loDelay = (unsigned long int) (((100 - dutyCycle) / 100) * period);

  // I want the wave to run for 0.5 seconds, find out how many wave periods per second
  counter = (unsigned long int) (500000 / period); // x1000000 to adjust for useconds

  // generate wave indefinitely (until we learn escape)
  for( ; counter > 0; counter-- )
  {
    // turn output high on bits 7:6 
    *portOutK |= 0x40;

    // maintain for the period of the wave
    myDelay( hiDelay, TRUE );

    // turn output low on bits 7:6
    *portOutK &= 0xBF;

    // maintain for off period of wave
    myDelay( loDelay, TRUE );
  }

  // no return - void 
}


/*################################################   Program Functionality   ###*/
void keyEcho()
{
  // variables
  volatile unsigned char entry = 0;
  const char instructions[] =
    "\r\nPress any key to see it echoed. Press ESC to quit.\r\n";
  
  // print prompt message
  printString( instructions );
  
  // get a character
  entry = getch();
  
  // echo until escape is hit 
  while( entry != ESC )
  {
    // separate the entry and the echo
    printString( " -> " );

    // echo the character
    putch( entry );
    printString( ENDL );
    
    // get the next entry
    printString( instructions );
    entry = getch();
  }

  // no return - void  
}


void ASCIIecho()
{
  // variables
  volatile unsigned char entry = 0;
  const char instructions[] =
    "\r\nPress any key to see its HEX value echoed. Press ESC to quit.\r\n";
  
  // print prompt message
  printString( instructions );
  
  // get a character
  entry = getch();
  
  // echo until escape is hit 
  while( entry != ESC )
  {
    // print the entered value
    putch( entry );
    
    // separate the entry and the echo
    putch( ' ' );
    putch( '-' );
    putch( '>' );
    putch( ' ' );

    // echo the character
    putch( getFirstHex( entry ) );
    putch( getSecondHex( entry ) );
    printString( ENDL );
    
    // get the next entry
    printString( instructions );
    entry = getch();
  }

  // no return - void  
}


void setCursorSpeed( volatile int* cursorSpeed )
{
  // variables
  volatile unsigned char entry;
  volatile unsigned char settingChar;
  const char instructions1[] =
    "Enter + to increase the cursor rotation speed,\r\n";
  const char instructions2[] = 
    "      - to decrease,\r\n and escape to quit.\r\n";
    
  // print the prompt for an entry
  printString( instructions1 );
  printString( instructions2 );
  
  // report the current setting
  printString( "The current setting is " );
  putch( settingToChar( *cursorSpeed ) );
  printString( ENDL );
    
  // get the user's entry
  entry = getch();
  
  // keep adjusting the speed until the user presses escape
  while( entry != ESC )
  {
    // case: the user wants to decrease the cursor speed
    if( entry == '-' )
    {
      // case: the speed can be decreased
      if( *cursorSpeed > 0 )
      {
        // decrease the speed
        (*cursorSpeed)--;
      } 
    }
    // case: the user wants to increase the cursor speed
    else if( entry == '+' )
    {
      // case: the speed can be increased
      if( *cursorSpeed < ( NUM_CURSOR_SPEEDS - 1 ) )
      {
        // increase the speed
        (*cursorSpeed)++;
      } 
    }
    
    // report the current setting
    printString( "The current setting is now " );
    putch( settingToChar( *cursorSpeed ) );
    printString( ENDL );
    
    // get the next ccommand
    entry = getch();
  }
  
  // no return - void
}


void blinkLED()
{
  // variables
  volatile unsigned char entry = '$';
  volatile unsigned int setting = DEFAULT_SETTING;
  const char instructions1[] =
    "\r\nEnter + to increase the blink rate,\r\n";
  const char instructions2[] =  
    "      - to decrease,\r\n and escape to quit.\r\n";
    
  // print the prompt for an entry
  printString( instructions1 );
  printString( instructions2 );

  // report the current setting
  printString( "The current setting is " );
  putch( settingToChar( setting ) );
  printString( ENDL );
  
  // keep blinking until the user presses escape
  while( entry != ESC )
  {
    // turn the LED high, and wait
    *portOutB |= 0x80;
    myDelay( BLINK_DELAYS[ setting ], FALSE );
    
    // turn the LED off
    *portOutB &= 0x7F;
    myDelay( BLINK_DELAYS[ setting ], FALSE );
    
    // case: the user wants to decrease the blink rate
    if( entry == '-' )
    {
      // case: the blink rate can be increased
      if( setting > 0 )
      {
        // decrease the blink rate setting (increases delay)
        setting--;
      } 
    }
    // case: the user wants to increase the blink rate
    else if( entry == '+' )
    {
      // case: the rate can be decreased
      if( setting < ( NUM_BLINK_DELAYS - 1 ) )
      {
        // increase the blink rate setting (decreases delay)
        setting++;
      } 
    }
  
    // if the speed was changed, report the new speed and instructions
    if( entry != '$' )
    {
      // report the new speed
      printString( "\r\nThe current setting is now " );
      putch( settingToChar( setting ) );
      printString( ENDL );
      
      // give instructions
      printString( instructions1 );
      printString( instructions2 );
    }
  
    // prevent setting from being changed if no input is given
    entry = '$';
      
    // case: a key was hit (only look for input if it is given)
    if( kb_hit() )
    {
      // get the response
      entry = getch();
    }
  }

  // no return - void
}


void soundTone()
{
  // variables
  volatile unsigned char entry = '$';
  volatile unsigned int setting = DEFAULT_SETTING;
  const char instructions1[] =
    "\r\nEnter + to increase the frequency of the tone,\r\n";
  const char instructions2[] =  
    "      - to decrease,\r\n and escape to quit.\r\n";
    
  // print the prompt for an entry
  printString( instructions1 );
  printString( instructions2 );
  
  // report the default tone
  printString( "The current tone is: " );
  putch( ( ( setting & 0xFF ) + 0x41 ) );
  printString( ENDL );
  
  // keep sounding tone until the user presses escape
  while( entry != ESC )
  {
    // pulse the tone
//    genRectangleWavePulse( TONE_PERIODS[ setting ] , 50 );
    
    // case: the user wants to decrease the pitch of the tone
    if( entry == '-' )
    {
      // case: the pitch can be decreased
      if( setting > 0 )
      {
        // decrease the pitch
        setting--;
      } 
    }
    // case: the user wants to increase the pitch of the tone
    else if( entry == '+' )
    {
      // case: the pitch can be increased
      if( setting < ( NUM_TONE_PERIODS - 1 ) )
      {
        // increase the pitch
        setting++;
      } 
    }
    
    // if the setting was changed, report the new tone
    if( entry != '$' )
    {
      // report the new tone setting
      printString( "The current tone is: " );
      putch( ( ( setting & 0xFF ) + 0x41 ) );
      printString( ENDL );
    }
    
    // prevent setting changes without input
    entry = '$';
    
    // case: a key was hit
    if( kb_hit() )
    {
      // get the response
      entry = getch();
    }
  }

  // no return - void
}

