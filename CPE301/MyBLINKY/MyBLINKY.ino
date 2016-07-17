/*
Creator:      Terence Henriod
Course:       CPE301
Section:      1101
Program Name: MyBLINKY
Revision #:   v0.01
Date:         9/29/2013


ORIGINAL PROGRAM:
=================
Blink without Delay
 
 Turns on and off a light emitting diode(LED) connected to a digital  
 pin, without using the delay() function.  This means that other code
 can run at the same time without being interrupted by the LED code.
 
 The circuit:
 * LED attached from pin 13 to ground.
 * Note: on most Arduinos, there is already an LED on the board
 that's attached to pin 13, so no hardware is needed for this example.
 
 
 created 2005
 by David A. Mellis
 modified 8 Feb 2010
 by Paul Stoffregen
 
 This example code is in the public domain.

 
 http://www.arduino.cc/en/Tutorial/BlinkWithoutDelay


=====================================================
   MyBLINKY

The base source code was provided by the Russell Introduction to Embedded
Systems text. It was then modified for the purposes of the laboratory
assignment.

MyBLINKY causes the LED built into the Arduino Mega board to blink on and
off.

It is important to note that the LED is controlled by pin 7 of the portB
register.


=====================================================

 */


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     Constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

const unsigned long int SHORT_DELAY = 150;    // in milliseconds
const unsigned long int MID_DELAY = 300;     // in milliseconds
const unsigned long int LONG_DELAY = 1500;    // in milliseconds

const unsigned char PIN_7_HI = 0x80;
const unsigned char PIN_7_LO = 0x7F;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     Function Prototypes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void myDelay( unsigned long mSecondsAprox );

void tester( unsigned char* port );

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void setup()
{
  // get a pointer to set the data direction register for port B
  unsigned char* portDDRB;
  
  // the address of port B on the Arduino Mega is register 24h
  portDDRB = (unsigned char *) 0x24;
  
  // bit 7 in the port B DataDirectionRegister determines if pin/bit 7 in portB
  // is input or output. by giving the bit in the DDR a value of 1, we make
  // pin 7 in port B an output pin.
  // by ORing the value currently in the register with a value of one, we can
  // ensure that the bit contains a value of 1. We have to do this bitwise, but
  // using entire bytes so we are ORing the bits in register 24h with the hex
  // equivalent of 1000 0000 (so that we have a 1 at bit 7 but won't alter other
  // bits)
  *portDDRB |= 0x80;
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void loop()
{
  // create a pointer variable to reference the LED pin/port
  unsigned char* portB;
  
  // the address of the port B register is 25h
  portB = (unsigned char *) 0x25;

/* 

  THIS CODE IS THE ORIGINAL CODE (MORE OR LESS). 
  THE DOCUMENTATION I HAVE ADDED IS VERY VALUABLE. DO NOT DELETE THIS COMMENT. 
  """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
  // use the bitwise OR assignment (ex: 11001 |= 00110 results in 11111)
  // we are changing the data sent to the LED to high using a bitwise
  // operation that will set bit 7 (1000 0000)to high  
  *portB |= 0x80;   // this is not an address, but an actual value
  
  // delay the program to leave the LED at its current state
  myDelay( SHORT_DELAY );
  
  // use the bitwise AND assignment (ex: 110101 &= 001110 results in 000100)
  // we are assigning the LED a low value (0111 1111). This bit string will
  // set other bits to zero or leave them alone since we are ANDing
  *portB &= 0x7F;
  
  // delay the program to leave the LED at its current state
  myDelay( LONG_DELAY); 
*/

  // initialize LED to low
  *portB &= PIN_7_LO;
  
  // perform two rapid blinks
  *portB |= PIN_7_HI;
  myDelay( SHORT_DELAY );
  *portB &= PIN_7_LO;
  myDelay( SHORT_DELAY );
  *portB |= PIN_7_HI;
  myDelay( SHORT_DELAY );
  
  
  // perform a long blink
  *portB &= PIN_7_LO; 
  myDelay( SHORT_DELAY);
  *portB |= PIN_7_HI;
  myDelay( MID_DELAY );
  *portB &= PIN_7_LO;


  tester( portB );

}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     Function Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void myDelay( unsigned long mSecondsAprox )
{
  // create a dummy variable, used to delay the program execution via
  // repeated looping
  volatile unsigned long ticker;

  // get a value for the amount of time to wait  
  unsigned long endTime = MID_DELAY * mSecondsAprox;
  
  // delay the program for the specified time by simply reassigning i 
  // many times
  for( ticker = 0; ticker < endTime; ticker ++ );
}


void tester( unsigned char* port )
{
  // set LED pin to low
  *port &= PIN_7_LO;
  
  // wait long time
  myDelay( LONG_DELAY );
  
  // set LED pin to high 
  *port |= PIN_7_HI;
  
  // wait long time
  myDelay( LONG_DELAY );
  
  // LED back to low
  *port &= PIN_7_LO;
  
  // no return - void
}







/*

// constants won't change. Used here to 
// set pin numbers:
const int ledPin =  13;      // the number of the LED pin

// Variables will change:
int ledState = LOW;             // ledState used to set the LED
long previousMillis = 0;        // will store last time LED was updated

// the follow variables is a long because the time, measured in miliseconds,
// will quickly become a bigger number than can be stored in an int.
long interval = 5000;           // interval at which to blink (milliseconds)

void setup() {
  // set the digital pin as output:
  pinMode(ledPin, OUTPUT);      
}

void loop()
{
  // here is where you'd put code that needs to be running all the time.

  // check to see if it's time to blink the LED; that is, if the 
  // difference between the current time and last time you blinked 
  // the LED is bigger than the interval at which you want to 
  // blink the LED.
  unsigned long currentMillis = millis();
 
  if(currentMillis - previousMillis > interval) {
    // save the last time you blinked the LED 
    previousMillis = currentMillis;   

    // if the LED is off turn it on and vice-versa:
    if (ledState == LOW)
      ledState = HIGH;
    else
      ledState = LOW;

    // set the LED with the ledState of the variable:
    digitalWrite(ledPin, ledState);
  }
}
*/
