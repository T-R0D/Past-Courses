/*=============================================================================
 Creator:      Terence Henriod
 Course:       CPE301
 Section:      1101
 Program Name: Wave Generation
 Revision #:   v0.01
 Date:         10/15/2013
 
 ==============================================================================*/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Global Variables (Yes, these are global!)
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Constants
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
 
 const int DEBUGGING = 1;
const unsigned long int SHORT_DELAY = 150;    // in milliseconds
const unsigned long int MID_DELAY = 300;     // in milliseconds
const unsigned long int LONG_DELAY = 1500;    // in milliseconds


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Function Prototypes
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


void genRectangleWave( unsigned long int frequencyHz, unsigned int dutyCycle );

void MyMicroDelay( unsigned long int uSeconds );

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Setup
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void setup()
{
  // variables
  volatile unsigned char* portDDRB = (unsigned char*) 0x24;   // portB DDR (LED)
  *portDDRB |= 0xC0;                         // set pins 7:6 to output (1100 0000)

  /*========= FOR TIMER 1 =====================================================*/
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
  // DO NOT use address 0x16!!!
  *tc1Flag |= 0x01;  // write 1 to clear flag (seriously)
  /*========= END TIMER 1 =====================================================*/



  //Initialize serial and wait for port to open:
  Serial.begin(9600); 
  while (!Serial)
  {
    // wait for serial port to connect. Needed for Leonardo only
  }


  // no return - void setup
}



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Loop
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void loop()
{
  // generate a 440Hz wave
  genRectangleWave( 200, 50 );
  
  // delay to differentiate between on and off
  MyMicroDelay( 30000 );

  // end of loop function - restart from main
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Function Implementations
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void genRectangleWave( unsigned long int frequencyHz, unsigned int dutyCycle )
{
  // variables
  volatile unsigned char* portB = (unsigned char*) 0x25;  // for B7
    *portB &= 0x7F;  // ensure signal is low
  volatile unsigned long int period = 0;
  volatile unsigned long int hiDelay = 0;
  volatile unsigned long int loDelay = 0;
  volatile unsigned long int counter = 0;

  // find the period of the wave
  // if the wave vibrates at frequency x, the period is x^-1
  // then x1000000 to convert to useconds conversion
  period = (unsigned long int)( 1000000.0 / frequencyHz );

  // scale the high and low periods according to the duty cycle
  hiDelay = (unsigned long int) ((dutyCycle / 100.0) * period);  
  loDelay = (unsigned long int) (((long) (100 - dutyCycle) / 100.0) * (long )period);

  // I want the wave to run for 3 seconds, find out how many wave periods per second
  counter = (unsigned long int) (3000000 / period); // x1000000 to adjust for useconds

  // generate wave indefinitely (until we learn escape)
  while( counter >= 0 )
  {
    // generate wave indefinitely (until we learn escape)
    // turn output high 
    *portB |= 0x80;

    // maintain for the period of the wave
    MyMicroDelay( hiDelay );

    // turn output low
    *portB &= 0x7F;   

    // maintain for off period of wave
    MyMicroDelay( loDelay );
    MyMicroDelay( 32000 );
    
    // decrement the counter
    counter = counter - 1;
  }

  // no return - void 
}


void MyMicroDelay( unsigned long int uSeconds )
{
  // variables
  // TimerCouNTer1 registers
  volatile unsigned int* tcRegisters;   // unsigned int is assumed to be 16 bit
  tcRegisters = (unsigned int*) 0x84;
  volatile unsigned char* tcc1A = (unsigned char*) 0x80;   // timer/counter1 control A
  volatile unsigned char* tcc1B = (unsigned char*) 0x81;   // timer/counter1 control B
  volatile unsigned char* tc1Flag = (unsigned char*)0x36;   // contains TOV1 flag
  volatile unsigned int timerStartVal = 0;
  volatile unsigned int countupVal = 0;

  // ensure the clock is stopped
  *tcc1B = 0;  // OK because normal mode

  // calculate the number of ticks we need to wait
  /*
    A good microseconds pre-scale is 8 because it will allow us to use up to 32768 uSeconds
   
   After the conversion, we see that 0.5 is the number of ticks per microsecond  
   */
  // get countup value
  countupVal = (unsigned int) ( uSeconds * 0.5 );

  // calculate timer start value (MAX + 1 - countupVal)
  timerStartVal = (unsigned int) (65536 - (long) countupVal);

  // setup for timing
  // store the calculated timer start value         
  //TODO: Use the 16 bit type defined in the header file
  *tcRegisters = timerStartVal;

  // clear the TOV flag
  *tc1Flag |= 0x01;

  // start timing
  // start the timer to count in the 8 pre-scale
  // (set CS2, CS1, CS0)
  *tcc1B = 0x02;  // OK for normal mode

  // let TCNT1 count up to max (that is, TOV1 flag gets set)
  while( (*tc1Flag & 0x01) == 0x00 )
  {
  }

  // clean up - stop counter
  // set the clock select bits for clock stopping
  *tcc1B = 0;   // OK for normal mode

  // clear the TOV1 flag
  *tc1Flag |= 0x01;

  // no return - void 
}
