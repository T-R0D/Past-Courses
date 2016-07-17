/*=============================================================================
Creator:      Terence Henriod
Course:       CPE301
Section:      1101
Program Name: MyBLINKY
Revision #:   v0.01
Date:         10/15/2013


==============================================================================*/


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     Constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#define DEBUGGING 0

const unsigned long int SHORT_DELAY = 150;    // in milliseconds
const unsigned long int MID_DELAY = 300;     // in milliseconds
const unsigned long int LONG_DELAY = 1500;    // in milliseconds

const unsigned char PIN_7_HI = 0x80;
const unsigned char PIN_7_LO = 0x7F;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     Function Prototypes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void ThreeStepBlink( volatile unsigned char* port );

void MyDelay( double mSeconds );

void EgbertDelay( unsigned long mSeconds );


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void setup()
{
  // variables
  volatile unsigned char* portDDRB = (unsigned char*) 0x24;   // portB DDR (LED)
    *portDDRB |= 0x80;                                      // set pin 7 to output
  volatile unsigned char* tcc1A = (unsigned char*) 0x80;   // timer/counter1 control A
    *tcc1A = 0;                                         // set all bits to 0 for normal mode
                                                        // specifically WGM11 and WGM10
  volatile unsigned char* tcc1B = (unsigned char*) 0x81;   // timer/counter1 control B
    *tcc1B = 0;                                         // set all bits to 0 for normal mode
                                                        // specifically WGM13 and WGM11
                                                        // and CS2:0 for clock stop
  volatile unsigned char* tcc1C = (unsigned char*) 0x82;   // timer/counter1 control C
    *tcc1C = 0;                                            // ensure register flags are cleared
  volatile unsigned char* timsk1 = (unsigned char*) 0x6F;  // interrupt mask register
    *timsk1 = 0;                                           // ensure these ar cleared too
  volatile unsigned char* tc1Flag = (unsigned char*) 0x36;  // timer/counter interrupt flag register
                                                            // NOT 0x16!!!
    *tc1Flag |= 0x01;  // write 1 to clear flag (seriously)
 
#if DEBUGGING
 //Initialize serial and wait for port to open:
  Serial.begin(9600); 
  while (!Serial)
  {
    // wait for serial port to connect. Needed for Leonardo only
  }
#endif  

  // no return - void setup
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void loop()
{
  // create a pointer variable to reference the LED pin/port, make volitile
  // to prevent compiler optimization
    // the address of the port B register is 25h
  volatile unsigned char* portB = (unsigned char*) 0x25;

  // initialize LED to low
  *portB &= 0x7F;
  
  // perform desired blink operation
  ThreeStepBlink( portB );
  
  // end of loop - restart
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     Function Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


void ThreeStepBlink( volatile unsigned char* port )
{
  // start with the LED low
  *port &= 0x7F;  
  
  // perform two rapid blinks
  *port |= PIN_7_HI;
  MyDelay( SHORT_DELAY );
  *port &= PIN_7_LO;
  MyDelay( SHORT_DELAY );
  *port |= PIN_7_HI;
  MyDelay( SHORT_DELAY );
  
  
  // perform a long blink
  *port &= PIN_7_LO; 
  MyDelay( SHORT_DELAY);
  *port |= PIN_7_HI;
  MyDelay( MID_DELAY );
  *port &= PIN_7_LO;

  // no return - void
}


void MyDelay( double mSeconds )
{
  // variables
  // TimerCouNTer1 registers
  volatile unsigned char* tcnt1H = (unsigned char*) 0x85; // more significant, 15:8
  volatile unsigned char* tcnt1L = (unsigned char*) 0x84; // 7:0
  // tcRegisters replaces the two above at the same time
  volatile unsigned int* tcRegisters;   // unsigned int is assumed to be 16 bit
    tcRegisters = (unsigned int*) 0x84;
  volatile unsigned char* tcc1A = (unsigned char*) 0x80;   // timer/counter1 control A
  volatile unsigned char* tcc1B = (unsigned char*) 0x81;   // timer/counter1 control B
  volatile unsigned char* tc1Flag = (unsigned char*)0x36;   // contains TOV1 flag
  volatile unsigned int timerStartVal = 0;
  volatile unsigned int countupVal = 0;
  
  // ensure the clock is stopped
  // set the clock select bits for clock stopping
  // i.e. CS02, CS01, CS00 to (000)
  *tcc1B &= 0xF8;   // (1111 1000)
  
  // calculate the number of ticks we need to wait
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
    // get countup value
    countupVal = (unsigned int) ( mSeconds * 15.625);

    // calculate timer start value (MAX + 1 - countupVal)
    timerStartVal = (unsigned int) (65536 - (long) countupVal);

  // setup for timing
    // store the calculated timer start value         
    //TODO: Use the 16 bit type defined in the header file
    *tcRegisters = timerStartVal;                                         // 124? WTF?
    
    // clear the TOV flag
    *tc1Flag |= 0x01;
  
  // start timing
    // start the timer to count in the 1024 pre-scale
    // (set CS2, CS1, CS0 to 101 (XXXX X101) )
    *tcc1B = 0x05;
    
    // let TCNT1 count up to max (that is, TOV1 flag gets set)
    while( (*tc1Flag & 0x01) == 0x00 )
    {}

  // clean up - stop counter
    // set the clock select bits for clock stopping
    // i.e. CS02, CS01, CS00 to (000)
    *tcc1B = 0;   // (1111 1000)    
    
    // clear the TOV1 flag
    *tc1Flag |= 0x01;
    
  // no return - void 
}


void EgbertDelay( unsigned long mSeconds )
{
  volatile unsigned int* myTCNT1 = (unsigned int*) 0x84;
  volatile unsigned char* myTCCR1B = (unsigned char*) 0x81;
  volatile unsigned char* myTIFR1 = (unsigned char *) 0x36;
  
  *myTCNT1 = (unsigned int) (65536 - (long) (15.625 * mSeconds));
  *myTCCR1B = 0b00000101;
  while( (*myTIFR1 & 0x01) == 0 )
  {}
 *myTCCR1B = 0;
 *myTIFR1 &= 0x01; 
}



