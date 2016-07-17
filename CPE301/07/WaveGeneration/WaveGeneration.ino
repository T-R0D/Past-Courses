/*=============================================================================
 Creator:      Terence Henriod
 Course:       CPE301
 Section:      1101
 Program Name: Wave Generation
 Revision #:   v0.01
 Date:         10/15/2013
 
 ==============================================================================*/


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

void openShutter( unsigned int speedCode );

void MyDelay( unsigned long int mSeconds );

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

  if( DEBUGGING )
  {
    //Initialize serial and wait for port to open:
    Serial.begin(9600); 
    while (!Serial)
    {
      // wait for serial port to connect. Needed for Leonardo only
    }
    Serial.println( "!!!!!!!!!!!!!!!!!!!!!!" );
    Serial.println( "!We are in debug mode!" );
    Serial.println( "!!!!!!!!!!!!!!!!!!!!!!" );
  }  

  // no return - void setup
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Loop
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void loop()
{

  // generate 16 kHz wave
  //genRectangleWave( 16000, 50 );

  // generate the 400Hz 'A' tone
  //genRectangleWave( 440, 50 );

  // generate the 500Hz wave with 30% duty cycle
  //genRectangleWave( 500, 30 );

  // test the shutter timer with speedCode 1
  //openShutter( 1 );

  // test the shutter timer with speedCode 5
  openShutter( 5 );

  // test the shutter timer with speedCode 10
  //openShutter( 10 );

  // delay functionality to make difference between loop calls noticeable
  MyDelay( 2000 );

  // end of loop function - restart from main
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Function Implementations
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void genRectangleWave( unsigned long int frequencyHz, unsigned int dutyCycle )
{
  // variables
  volatile unsigned char* portB = (unsigned char*) 0x25;  // for B6
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
  hiDelay = (unsigned long int) ((dutyCycle / 100) * period);  
  loDelay = (unsigned long int) (((100 - dutyCycle) / 100) * period);

  // I want the wave to run for 3 seconds, find out how many wave periods per second
  counter = (unsigned long int) (3000000 / period); // x1000000 to adjust for useconds

  // generate wave indefinitely (until we learn escape)
  for( ; counter > 0; counter-- )
  {
    // generate wave indefinitely (until we learn escape)
    // turn output high 
    *portB |= 0x40;

    // maintain for the period of the wave
    MyMicroDelay( hiDelay );

    // turn output low
    *portB &= 0xBF;

    // maintain for off period of wave
    MyMicroDelay( loDelay );
  }

  // no return - void 
}


void openShutter( unsigned int speedCode )
{
  // variables
  volatile unsigned char* portB = (unsigned char*) 0x25;  // for B7
  *portB &= 0x7F;  // ensure shutter is closed
  volatile unsigned long int openTime = 0;  // in milliseconds

  // determine how long to open the shutter based on the chosen code
  switch( speedCode )
  {
    // shutter speed 0
  case 0:
    // we have decided in this case that the shutter will be open
    // for 1 second. Find the period in milliseconds
    openTime = 1000;
    break; 

    // shutter speed 1
  case 1:
    // we have decided in this case that the shutter will be open
    // for 0.5 second. Find the period in milliseconds
    openTime = 500;
    break; 

    // shutter speed 2
  case 2:
    // we have decided in this case that the shutter will be open
    // for 0.25 second. Find the period in milliseconds
    openTime = 250;
    break; 

    // shutter speed 3
  case 3:
    // we have decided in this case that the shutter will be open
    // for 0.125 second. Find the period in milliseconds
    openTime = 125;
    break; 

    // shutter speed 4
  case 4:
    // we have decided in this case that the shutter will be open
    // for 1/15 (o.067) second. Find the period in milliseconds
    openTime = 67;
    break; 

    // shutter speed 5
  case 5:
    // we have decided in this case that the shutter will be open
    // for 1/30 (0.0333) second. Find the period in milliseconds
    openTime = 33;
    break; 

    // shutter speed 6
  case 6:
    // we have decided in this case that the shutter will be open
    // for 1/60 (0.0167) second. Find the period in milliseconds
    openTime = 17;
    break; 

    // shutter speed 7
  case 7:
    // we have decided in this case that the shutter will be open
    // for 1/125 (0.008) second. Find the period in milliseconds
    openTime = 8;
    break; 

    // shutter speed 8
  case 8:
    // we have decided in this case that the shutter will be open
    // for 1/250 second. Find the period in milliseconds
    openTime = 4;
    break;

    // shutter speed 9
  case 9:
    // we have decided in this case that the shutter will be open
    // for 1/500 second. Find the period in milliseconds
    openTime = 2;
    break;

    // shutter speed 10
  case 10:
    // we have decided in this case that the shutter will be open
    // for 1/1000 second. Find the period in milliseconds
    openTime = 1;
    break;
  }

  // open the shutter
  *portB |= 0x80;

  // leave shutter open for specified time
  MyDelay( openTime );

  // close the shutter
  *portB &= 0x7F;

  // no return - void
}


void MyDelay( unsigned long int mSeconds )
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
  {
  }

  // clean up - stop counter
  // set the clock select bits for clock stopping
  // i.e. CS02, CS01, CS00 to (000)
  *tcc1B = 0;   // (1111 1000)    

  // clear the TOV1 flag
  *tc1Flag |= 0x01;

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
    A good microseconds pre-scale is 1 because it will allow us to use up to 4096 uSeconds
   
   After the conversion, we see that 0.0625 is the number of ticks per microsecond  
   */
  // get countup value
  countupVal = (unsigned int) ( uSeconds * 0.0625 );

  // calculate timer start value (MAX + 1 - countupVal)
  timerStartVal = (unsigned int) (65536 - (long) countupVal);

  // setup for timing
  // store the calculated timer start value         
  //TODO: Use the 16 bit type defined in the header file
  *tcRegisters = timerStartVal;                                         // 124? WTF?

  // clear the TOV flag
  *tc1Flag |= 0x01;

  // start timing
  // start the timer to count in the  pre-scale
  // (set CS2, CS1, CS0)
  *tcc1B = 0x00;  // OK for normal mode

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



