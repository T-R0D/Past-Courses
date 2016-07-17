void MyDelay( unsigned long mSeconds );

void setup()
{
  // variables
  volatile unsigned char* portDDRB = (unsigned char*) 0x24;   // portB DDR (LED)
    *portDDRB |= 0x80;
  volatile unsigned char* tcc1A = (unsigned char*) 0x80;   // timer/counter1 control A
    *tcc1A = 0;
  volatile unsigned char* tcc1B = (unsigned char*) 0x81;   // timer/counter1 control B
    *tcc1B = 0;
  volatile unsigned char* tcc1C = (unsigned char*) 0x82;   // timer/counter1 control B
    *tcc1C = 0;
  volatile unsigned char* timsk1 = (unsigned char*) 0x6F;
    *timsk1 = 0;
  volatile unsigned char* tc1Flag = (unsigned char*) 0x36;  // timer/counter interrupt flag register
    *tc1Flag = 1;                                           // NOT 0x16!!!

}

void loop ()
{
volatile unsigned char* portB = (unsigned char*) 0x25;
  *portB |= 0x80;
  
MyDelay( 500 );
*portB &= 0x7F;
MyDelay( 1000 );
}


void MyDelay( unsigned long mSeconds )
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
