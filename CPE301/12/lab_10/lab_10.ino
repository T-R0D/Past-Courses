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
const int DEFAULT_TIMER_PRE = 0x05;
const int DEFAULT_TIMER_ONE = 48911;                      // sets timer 1s delay
const int TCNT_MAX = 0xFFFF;                              // MAX counter val
const int DEBOUNCE = 20;                                  // miliseconds
const int NUM_CURSOR_SPEEDS = 5;
const int CURSOR_DELAYS[ NUM_CURSOR_SPEEDS ] =
  { 54000, 48000, 32000, 16000, 8000 };                   // in syswrk() calls
const int NUM_BLINK_DELAYS = 10;
const int BLINK_DELAYS[ NUM_BLINK_DELAYS ] =
  { 1000, 666, 500, 333, 250, 125, 100, 75, 50, 25 };     // in milliseconds
const int NUM_TONE_PERIODS = 12;
const int TONE_FREQ[ NUM_TONE_PERIODS] =
  { 440, 466, 494, 523, 554, 587, 624, 659, 698, 740, 784, 831 };  // in Hz

/*~~~~~   Character Values   ~~~~~*/
const unsigned char G_INTERRUPT_MASK = 0x80;
const unsigned char BS = 0x08;
const unsigned char ESC = 0x1B;

/*~~~~~   Necessary Strings   ~~~~~*/

/*~~~~~   LAB 10 CRAP   ~~~~~*/
const int NOTE_PRE = 001;     // seconds per tick = 1/16000000
const int MID_C_T_START = 50240;// start value = *half* period / secPerTick
const int NUM_NOTES = 12;
// I did calculations to hard code to minimize global var use
const int NOTE_STARTS[NUM_NOTES] =
  { 47354, 48368, 49341, 50240, 51095, 51907, 52715, 53396,
    54074, 54725, 55331, 55909 };
  
volatile static unsigned int gNoteSetting = 5;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Global Variables (Yes, these are global!)
 Mostly used for register pointers that can't be declared constant.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*~~~~~   AVR Status Register   ~~~~~*/
volatile unsigned char* AVR_stat = (unsigned char*) 0x5F;

/*~~~~~   GPIO Registers   ~~~~~*/
volatile unsigned char* portInB  = (unsigned char*) 0x23;
volatile unsigned char* DDR_B    = (unsigned char*) 0x24;
volatile unsigned char* portOutB = (unsigned char*) 0x25;

volatile unsigned char* portInK  = (unsigned char*) 0x106;
volatile unsigned char* DDR_K    = (unsigned char*) 0x107;
volatile unsigned char* portOutK = (unsigned char*) 0x108;

/*~~~~~   Timer/Counter Registers   ~~~~~*/
volatile unsigned char* myTIMSK1 = (unsigned char*) 0x6F; // Interrupt Mask
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

/*~~~~~   ISRs   ~~~~~*/
ISR( TIMER1_OVF_vect );

/*~~~~~   Essential Functions   ~~~~~*/
unsigned char kb_hit();
unsigned char getch();
void putch( unsigned char data );

/*~~~~~   String Functions   ~~~~~*/

/*~~~~~   Operational Functions   ~~~~~*/

/*~~~~~   Program Functionality Support Functions   ~~~~~*/

/*~~~~~   Program Functionality   ~~~~~*/


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void setup()
{
  // initialize interrupts globally
  *AVR_stat |= G_INTERRUPT_MASK;
 
  // initialize timer/counter1
  tc1init();
  
  // initialize the USART
  USART0init( BAUD_RATE );

  // initialize GPIO sound output
  *DDR_K    |= 0xF0;  // use upper nibble of K for sound output
  *portOutK &= 0x0F;  // turn sound off to start

  // initialize LED output
  *DDR_B    |= 0x40;   // portB6
  *portOutB &= 0;   // turn LED off to start

  // no return - void
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void loop()
{
  // variables
  volatile unsigned char input = '$';
  putch( input );
  // start with everything off and prepped
  *myTCCR1B &= 0b11111000;
  *myTCNT1  = NOTE_STARTS[gNoteSetting];
  *portOutB &= 0b10111111;
  
  // don't start until user is ready
  while( input != 's' )
  {
    // case: the user pressed a key
    if( kb_hit() )
    {
      // get the input
      input = getch();
      putch( input );
    }   
  };
  
  // start playing the tone
  *portOutB |= 0x40;
  
  // start the timer
  *myTCCR1B |= NOTE_PRE;

  // while the user doesn't want the tone to stop, keep it going
  while( input != ESC )
  {
    // the ISR will continously restart the timer and play the frequency
    // while the user lets the program run
    
    // case: the user pressed a key
    if( kb_hit() )
    {
      // increment the setting
      gNoteSetting++;
    
      // if necessary start over at 0
      if( gNoteSetting >= NUM_NOTES )
      {
        gNoteSetting = 0; 
      }
      
      // get the input
      input = getch();
    } 
  }
  
  // stop the timer to ensure that interrupts don't happen later,
  // turn of the tone
  *myTCCR1B &= 0xF8;
  *portOutB &= 0xBF;
  
  // end of loop function - restart from main
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Function Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/*###########################################   Initialization Functions   ###*/
void tc1init()
{
  // ensure clock is stopped, setup normal mode, disable output compares,
  // enable the Timer Overflow Interrupt
  *myTCCR1B = 0b00000000;
             // 0000     to disconnect the Output Compare pins (OC1)
             //     00   reserved, write 0
             //       00 as part of Normal Mode WGM11:10(see TCCR1B for ohters)

  *myTCCR1A = 0b00000000;
             // 00       disable InputCaptureNoiseCanceler and ICEdgeSelect
             //   0      reserved, must be written 0
             //    00    for Normal Mode WGM13:12 (seeTCCR1A for the rest) 
             //      000 to stop the clock
             //  ========  FYI =============
             //      001 to run the timer/counter with no pre-scale (1)
             //      010 to run the timer/counter with a pre-scale of 8
             //      101 to run the timer/counter with a pre-scale of 1024 
  *myTIFR1 = 0b00101111;
            // 00       reserved
            //   1      to clear TC Input Capture Flag
            //    0     reserved
            //     111  to clear TC Output Compare C:A Match flags
            //        1 to clear TimerOVerflow flag
  *myTIMSK1 = 0b00000001;
             // 00       reserved, write 0
             //   0      disable Input Capture Interrupt
             //    0     reserved, write 0
             //     000  disable Output Compare Match Interrupts for C:A
             //        1 set the Overflow Interrupt Enable (OIE)

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
                                     // ex: ((1111 0000) >> 8) == (0000 1111)
  *myUBRR0L = ( baudRegVal & 0xFF ); // store only the low byte in low register
  
  // no return - void
}


/*###############################################################   ISRs   ###*/

ISR( TIMER1_OVF_vect )
{
  // stop TIMER1
  *myTCCR1B &= 0xF8; // to set bits 2:0 to 0
  
  // reload the timer
  *myTCNT1 = NOTE_STARTS[gNoteSetting];
  
  // clear TOV flag
  *myTIFR1 |= 0x01;
  
  // toggle portOutB6
  *portOutB ^= 0x40;
  
  // restart timer
  *myTCCR1B |= NOTE_PRE;
  
  // no return - ISR
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


/*###################################################   String Functions   ###*/


/*##############################################   Operational Functions   ###*/

/*############################   Program Functionality Support Functions   ###*/


/*################################################   Program Functionality   ###*/

