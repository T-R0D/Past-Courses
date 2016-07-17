/* Simple Serial ECHO script : Written by ScottC 03/07/2012 */

// Prototype
void waveGen( volatile unsigned long halfTimeuency );
void myDelay( unsigned long mSecondsApx );

/* Use a variable called byteRead to temporarily store
   the data coming from the computer */
byte byteRead;
volatile unsigned char *myTCCR1A; 
volatile unsigned char *myTCCR1B; 
volatile unsigned char *myTCCR1C; 
volatile unsigned char *myTIMSK1; 
volatile unsigned int *myTCNT1; 
volatile unsigned char *myTIFR1; 
volatile unsigned char *portB; 

void setup() {                
// Turn the Serial Protocol ON
  Serial.begin(9600);
  
// Initialize Timer1 pointers. 
myTCCR1A = (unsigned char *) 0x80; 
myTCCR1B = (unsigned char *) 0x81; 
myTCCR1C = (unsigned char *) 0x82; 
myTIMSK1 = (unsigned char *) 0x6F; 
myTCNT1 = (unsigned int *) 0x84; 
myTCCR1B = (unsigned char *) 0x81; 
myTIFR1 = (unsigned char *) 0x36; 
// Initialize Timer1 for NORMAL mode, off, and no interrupts. 
*myTCCR1A = 0; 
*myTCCR1B = 0; 
*myTCCR1C = 0; 
*myTIMSK1 = 0; 
// Initialize GPIO PortB 
volatile unsigned char *portDDRB; 
portDDRB = (unsigned char *) 0x24; 
*portDDRB |= 0x80; 

}

void loop() {
  
  float halfTime;
  
   /*  check if data has been sent from the computer: */
  if (Serial.available()) {
    /* read the most recent byte */
    byteRead = Serial.read();
    
    /*ECHO the value that was read, back to the serial port. */
    Serial.write(byteRead);
  }
  
  switch( byteRead )
  {
     case 'A': 
         halfTime = 1.136; 
         break;
     case 'a':
         halfTime = 1.073;
         break;
     case 'B':
         halfTime = 1.012;
         break;
     case 'C':
         halfTime = 0.956;
         break;
     case 'c':
         halfTime = 0.903;
         break;
     case 'D':
         halfTime = 0.852;
         break;
     case 'd':
         halfTime = 0.801;
         break;
     case 'E':
         halfTime = 0.759;
         break;
     case 'F':
         halfTime = 0.716;
         break;
     case 'f':
         halfTime = 0.676;
         break;    
     case 'G':
         halfTime = 0.638;
         break;
     case 'g':
         halfTime = 0.602;
         break;
     default:
         halfTime = 0;
         break;    
  }

    volatile unsigned char* portB = (unsigned char*) 0x25;
  *portB &= 0xBF;

  if( halfTime != 0 )
  {
      // Calculate period
     //period = 1 / halfTimeuency;
     
     // Set low
     *portB &= 0xBF;
     
     // Delay for half period
     
       *myTCNT1 = (unsigned int) (65536 - (long) (15.625 * halfTime ) ); 
  
  // incorporate prescale 
  *myTCCR1B = 0x05; 
  
  // loop until the TOV flag is reached 
  while( (*myTIFR1 & 0x01 ) == 0 ); 
  
  // turn off the timer 
  *myTCCR1B = 0; 
  
  // write a 1 for whatever reason.. 
  *myTIFR1 |= 0x01;   
  

  
     
     // Set high
     *portB |= 0x40;
     
     // Delay for half period
   
            *myTCNT1 = (unsigned int) (65536 - (long) (15.625 * halfTime ) ); 
  
  // incorporate prescale 
  *myTCCR1B = 0x05; 
  
  // loop until the TOV flag is reached 
  while( (*myTIFR1 & 0x01 ) == 0 ); 
  
  // turn off the timer 
  *myTCCR1B = 0; 
  
  // write a 1 for whatever reason.. 
  *myTIFR1 |= 0x01;   
  }
  
 
  
}

void myDelay( unsigned long mSecondsApx )
{
 
  // calculate duration
  // we are multiplying the mSecondsApx with 15.625
  // this value was calculated in class and corresponds to 
  // how many ticks occur per millisecond. This value is 
  // used to load the timer. 
  *myTCNT1 = (unsigned int) (65536 - (long) (15.625 * mSecondsApx ) ); 
  
  // incorporate prescale 
  *myTCCR1B = 0x05; 
  
  // loop until the TOV flag is reached 
  while( (*myTIFR1 & 0x01 ) == 0 ); 
  
  // turn off the timer 
  *myTCCR1B = 0; 
  
  // write a 1 for whatever reason.. 
  *myTIFR1 |= 0x01;   
}

void waveGen( volatile unsigned long frequency )
{
  /*
   
  // Initialize variables 
  float period; 
  volatile unsigned char* portB = (unsigned char*) 0x25;
  *portB &= 0xBF;

  if( frequency != 0 )
  {
      // Calculate period
     //period = 1 / frequency;
     //Serial.println( period );
     
     // Set low
     *portB &= 0xBF;
     
     // Delay for half period
     
     myDelay( 1.13636 );
     //myDelay( 1000 / (2 * frequency) );
     
     // Set high
     *portB |= 0x40;
     
     // Delay for half period
     myDelay( 1.13636 ); 
  }*/
}
