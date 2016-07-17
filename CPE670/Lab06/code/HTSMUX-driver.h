/*
 * $Id$
 */

#ifndef __HTSMUX_DRIVER_H__
#define __HTSMUX_DRIVER_H__

/**
 * HTSMUX-driver.nxc provides an API for the HiTechnic Sensor MUX.  This program
 * demonstrates how to use that API.
 *
 * Changelog:
 * - 0.1: Initial release
 * - 0.2: ADDED 
 * - int smuxSensorLegoUS(const byte muxport);
 * - int smuxSensorLegoLightRaw(const byte muxport);
 * - int smuxSensorLegoLightNorm(const byte muxport);
 * - bool smuxSetSensorLegoLight(const byte muxport, const bool active);
 * - int smuxSensorLegoSoundRaw(const byte muxport);
 * - int smuxSensorLegoSoundNorm(const byte muxport);
 * - bool smuxSetSensorLegoSound(const byte muxport, const bool dba);
 * - REMOVED:
 * - int sensorLegoUS(const byte muxport);
 * - 0.3: Added proper return code to HT Accelerometer function
 * - 0.4: Changed how sensor types are stored, now uses single dimension array
 * - 0.5: Added LEGO Touch Sensor support
 *
 * Credits:
 * - Big thanks to HiTechnic for providing me with the hardware necessary to write and test this.
 * - John Hansen for answering my technical and philosophical questions
 * - Hunter Chiou for testing and feedback
 * - Gus Jansson for testing and feedback
 * - Kate Craig-Wood for feedback and LEGO Touch function
 *
 * License: You may use this code as you wish, provided you give credit where it's due.
 * 
 * Xander Soldaat (mightor_at_gmail.com)
 * 15 April 2012
 * version 0.5
 */

/**
 * Example code

#include "HTSMUX-driver.nxc"

task main () {

  // Variable to hold measurement data from US sensor
  byte distance = 0;
  
  // configure the port for low speed I2C
  SetSensor(S1, SENSOR_LOWSPEED);
  
  // Tell the SMUX to start scanning its ports.
  // this takes 500ms
  if (!HTSMUXscanPorts(S1)) {
    // Scan failed, handle the error
  }
  
  // The sensor in this example is a LEGO Ultra Sound sensor connected
  // to the SMUX's 1st port (msensor_S1_1).
  // SMUX sensor ports use the following naming standard:
  // If the SMUX is connected to the NXT's S1 port, it starts with msensors_S1.
  // If the SMUX were to be connected to the S2 port, it would start with msensors_S2
  // 
  // The second part denotes the SMUX's port, which it has 4 of.
  // If the sensor was connected to the SMUX port 1 and the SMUX was connected to 
  // the NXT's S1 port, the smuxport would be written as msensor_S1_1.
  // If the sensor was connected to the SMUX port 3 with the SMUX still connected 
  // to S1, it would be written as msensor_S1_3
  
  while (true) {
    distance = sensorLegoUS(msensor_S1_1);
    TextOut(0, LCD_LINE2, "DISTANCE:", true);
    NumOut(0, LCD_LINE3, distance);
    Wait(100);
  }
}
  
*/
  

// SMUX port
#define SMUXPORT S1

// LEGO TOUCH Sensor
#define TOUCH_SENSOR_THRESHOLD 500

// Macros for coverting
#define SPORT(X)  X / 4         /*!< Convert tMUXSensor to sensor port number */
#define MPORT(X)  X % 4         /*!< Convert tMUXSensor to MUX port number */
#define smuxSensorType(X)       smuxSensor(SPORT(X),MPORT(X)) /*!< Fetch the sensor type */
#define smuxSensor(X,Y)         smuxSensorTypes[X*4+Y]

// Registers
#define HTSMUX_I2C_ADDR         0x10  /*!< HTSMUX I2C device address */
#define HTSMUX_COMMAND          0x20  /*!< Command register */
#define HTSMUX_STATUS           0x21  /*!< Status register */

#define HTSMUX_MODE             0x00  /*!< Sensor mode register */
#define HTSMUX_TYPE             0x01  /*!< Sensor type register */
#define HTSMUX_I2C_COUNT        0x02  /*!< I2C byte count register */
#define HTSMUX_I2C_DADDR        0x03  /*!< I2C device address register */
#define HTSMUX_I2C_MADDR        0x04  /*!< I2C memory address register */
#define HTSMUX_CH_OFFSET        0x22  /*!< Channel register offset */
#define HTSMUX_CH_ENTRY_SIZE    0x05  /*!< Number of registers per sensor channel */

#define HTSMUX_ANALOG           0x36  /*!< Analogue upper 8 bits register */
#define HTSMUX_AN_ENTRY_SIZE    0x02  /*!< Number of registers per analogue channel */

#define HTSMUX_I2C_BUF          0x40  /*!< I2C buffer register offset */
#define HTSMUX_BF_ENTRY_SIZE    0x10  /*!< Number of registers per buffer */

// Command fields
#define HTSMUX_CMD_HALT         0x00  /*!< Halt multiplexer command */
#define HTSMUX_CMD_AUTODETECT   0x01  /*!< Start auto-detect function command */
#define HTSMUX_CMD_RUN          0x02  /*!< Start normal multiplexer operation command */

// Status
#define HTSMUX_STAT_NORMAL      0x00  /*!< Nothing going on, everything's fine */
#define HTSMUX_STAT_BATT        0x01  /*!< No battery voltage detected status */
#define HTSMUX_STAT_BUSY        0x02  /*!< Auto-dected in progress status */
#define HTSMUX_STAT_HALT        0x04  /*!< Multiplexer is halted status */
#define HTSMUX_STAT_ERROR       0x08  /*!< Command error detected status */
#define HTSMUX_STAT_NOTHING     0xFF  /*!< Status hasn't really been set yet */

// Channel modes
#define HTSMUX_CHAN_NONE        0X00  /*!< Turn off all options for this channel */
#define HTSMUX_CHAN_I2C         0x01  /*!< I2C channel present channel mode */
#define HTSMUX_CHAN_9V          0x02  /*!< Enable 9v supply on analogue pin channel mode */
#define HTSMUX_CHAN_DIG0_HIGH   0x04  /*!< Drive pin 0 high channel mode */
#define HTSMUX_CHAN_DIG1_HIGH   0x08  /*!< Drive pin 1 high channel mode */
#define HTSMUX_CHAN_I2C_SLOW    0x10  /*!< Set slow I2C rate channel mode */


/*!< Sensor types as detected by SMUX */
const byte HTSMUXAnalogue     = 0x00;
const byte HTSMUXLegoUS       = 0x01;
const byte HTSMUXCompass      = 0x02;
const byte HTSMUXColor        = 0x03;
const byte HTSMUXAccel        = 0x04;
const byte HTSMUXIRSeeker     = 0x05;
const byte HTSMUXProto        = 0x06;
const byte HTSMUXColorNew     = 0x07;
const byte HTSMUXIRSeekerNew  = 0x09;
const byte HTSMUXSensorNone   = 0xFF;
  
/*!< Sensor and SMUX port combinations */
const byte msensor_S1_1 = 0;
const byte msensor_S1_2 = 1;
const byte msensor_S1_3 = 2;
const byte msensor_S1_4 = 3;
const byte msensor_S2_1 = 4;
const byte msensor_S2_2 = 5;
const byte msensor_S2_3 = 6;
const byte msensor_S2_4 = 7;
const byte msensor_S3_1 = 8;
const byte msensor_S3_2 = 9;
const byte msensor_S3_3 = 10;
const byte msensor_S3_4 = 11;
const byte msensor_S4_1 = 12;
const byte msensor_S4_2 = 13;
const byte msensor_S4_3 = 14;
const byte msensor_S4_4 = 15;


/*!< SMUX info */
byte smuxSensorTypes[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
byte smuxStatus[4] = {HTSMUX_STAT_NOTHING, HTSMUX_STAT_NOTHING, HTSMUX_STAT_NOTHING, HTSMUX_STAT_NOTHING};

// Prototypes

// HiTechnic Sensor MUX port scanning function.  This must be called before 
// any sensors can be polled
bool HTSMUXscanPorts(const byte smux);

// HiTechnic Accelerometer
bool smuxReadSensorHTAccel(const byte muxport, int &x, int &y, int &z);

// HiTechnic IR Seeker V2
bool smuxReadSensorHTIRSeeker2AC(const byte muxport, byte &dir, byte &s1, byte &s2, byte &s3, byte &s4, byte &s5);
int smuxSensorHTIRSeeker2ACDir(const byte muxport);
bool smuxReadSensorHTIRSeeker2DC(const byte muxport, byte &dir, byte &s1, byte &s2, byte &s3, byte &s4, byte &s5, byte &avg) ;
int smuxSensorHTIRSeeker2DCDir(const byte muxport);
bool smuxReadSensorHTIRSeeker(const byte muxport, byte &dir, byte &s1, byte &s2, byte &s3, byte &s4, byte &s5);
int smuxSensorHTIRSeekerDir(const byte muxport);

// HiTechnic Colour Sensor (V1/V2)
bool smuxReadSensorHTColor(const byte muxport, int &colnum, int &red, int &green, int &blue);
int smuxSensorHTColorNum(const byte muxport);

// HiTechnic Compass Sensor
int smuxSensorHTCompass(const byte muxport);

// HiTechnic EOPD Sensor
int smuxSensorHTEOPD(const byte muxport);
int smuxSensorRawHTEOPD(const byte muxport);
bool smuxSetSensorHTEOPD(const byte muxport, const bool bStandard);

// HiTechnic Gyro
int smuxSensorHTGyro(const byte muxport, const int offset);

// Standard LEGO sensors
int smuxSensorLegoUS(const byte muxport);
int smuxSensorLegoLightRaw(const byte muxport);
int smuxSensorLegoLightNorm(const byte muxport);
bool smuxSetSensorLegoLight(const byte muxport, const bool active);
int smuxSensorLegoSoundRaw(const byte muxport);
int smuxSensorLegoSoundNorm(const byte muxport);
bool smuxSetSensorLegoSound(const byte muxport, const bool dba);
bool smuxReadSensorLegoTouch(const byte muxport);
bool smuxSetSensorLegoTouch(const byte muxport);

// Analogue Sensor specific functions
int HTSMUXreadAnalogue (const byte muxport);
bool HTSMUXsetAnalogueInactive(const byte muxport);
bool HTSMUXsetAnalogueActive(const byte muxport);

// Digital Sensor specific functions
bool HTSMUXreadDigital (const byte muxport, byte &data[], const byte offset, const byte length);

// Internal HiTechnic Sensor MUX functions that should be not used directly
bool _HTSMUXsetMode(const byte muxport, const byte mode);
byte _HTSMUXreadStatus(const byte smux);
bool _HTSMUXsendCmd(const byte smux, const byte cmd);


 
/**
 * Send a command to the SMUX.
 *
 * command can be one of the following:
 * -HTSMUX_CMD_HALT
 * -HTSMUX_CMD_AUTODETECT
 * -HTSMUX_CMD_RUN
 *
 * Note: this is an internal function and should not be called directly
 * @param smux the SMUX port number
 * @param cmd the command to be sent to the SMUX
 * @return true if no error occured, false if it did
 */ 
bool _HTSMUXsendCmd(const byte smux, const byte cmd)  {
  byte sendMsg[];
  
  ArrayBuild(sendMsg, HTSMUX_I2C_ADDR, HTSMUX_COMMAND, cmd);

  // Send the command to the SMUX
  while (I2CCheckStatus(smux) == STAT_COMM_PENDING) Yield();
  I2CWrite(smux, 0, sendMsg);

  // if the HTSMUX_CMD_AUTODETECT command has
  // been given, wait 500 ms
  if (cmd == HTSMUX_CMD_AUTODETECT) {
    Wait(500);

  // Wait 50ms after SMUX is halted
  } else if (cmd == HTSMUX_CMD_HALT) {
    Wait(50);
  }
  
  // Set the status variable appropriately 
  switch(cmd) {
    case HTSMUX_CMD_HALT:
        smuxStatus[smux] = HTSMUX_STAT_HALT;
        break;
    case HTSMUX_CMD_AUTODETECT:
        smuxStatus[smux] = HTSMUX_STAT_BUSY;
        break;
    case HTSMUX_CMD_RUN:
        smuxStatus[smux] = HTSMUX_STAT_NORMAL;
        break;
  }
  
  while ((I2CCheckStatus(smux) == STAT_COMM_PENDING) && (I2CCheckStatus(smux) != NO_ERR)) Yield();
  return (I2CCheckStatus(smux) == NO_ERR);
}


/**
 * Read the value returned by the sensor attached the SMUX. This function
 * is for I2C sensors.
 * @param muxport the SMUX sensor port number
 * @param data array to hold values returned from SMUX
 * @param offset the offset used to start reading from
 * @param length the size of the I2C reply
 * @return true if no error occured, false if it did
 */
bool HTSMUXreadDigital (const byte muxport, byte &data[], const byte offset, const byte length) {
  byte sendMsg[];
  
  // Work-around for built-in macros that can't grok this.
  byte smux = SPORT(muxport);
  byte channel = MPORT(muxport);
  
  ArrayInit(data, 0, length);
  ArrayInit(sendMsg, 0, 2);
  
  // If we're in the middle of a scan, abort this call
  if (smuxStatus[smux] == HTSMUX_STAT_BUSY) {
    return false;
  // Are we ready to run
  } else if (smuxStatus[smux] != HTSMUX_STAT_NORMAL) {
    if (!_HTSMUXsendCmd(smux, HTSMUX_CMD_RUN))
      return false;
  }
	
  sendMsg[0] = HTSMUX_I2C_ADDR; // Address of SMUX
  // Buffer register
  sendMsg[1] = HTSMUX_I2C_BUF + (channel * HTSMUX_BF_ENTRY_SIZE) + offset;
  // Query the SMUX and read the response
  while (I2CCheckStatus(smux) == STAT_COMM_PENDING) Yield();    
  return I2CBytes(smux, sendMsg, length, data);
} 


/**
 * Read the value returned by the sensor attached the SMUX. This function
 * is for analogue sensors.
 * @param muxport the SMUX sensor port number
 * @return the value of the sensor or -1 if an error occurred.
 */
int HTSMUXreadAnalogue (const byte muxport) {
  byte sendMsg[2];
  byte readMsg[2];
  const byte count = 2;
  
  // Work-around for built-in macros that can't grok this.
  byte smux = SPORT(muxport);
  byte channel = MPORT(muxport);
    
  if (smuxStatus[smux] == HTSMUX_STAT_BUSY) {
    return false;
  // Are we ready to run
  } else if (smuxStatus[smux] != HTSMUX_STAT_NORMAL) {
    if (!_HTSMUXsendCmd(smux, HTSMUX_CMD_RUN))
      return false;
  }
  
  sendMsg[0] = HTSMUX_I2C_ADDR; // Address of SMUX
               // Buffer register
  sendMsg[1] = HTSMUX_ANALOG + (channel * HTSMUX_AN_ENTRY_SIZE);
  
  // Query the SMUX and read the response
  while (I2CCheckStatus(smux) == STAT_COMM_PENDING) Yield();  
  if (I2CBytes(smux, sendMsg, count, readMsg))
    return ((readMsg[0] & 0x00FF) * 4) + (readMsg[1] & 0x00FF);
  else
    return 0;
}


/**
 * Read the status of the SMUX
 *
 * The status byte is made up of the following bits:
 *
 * | D7 | D6 | D4 | D3 | D2 | D1 | D1 |
 * -D1 - HTSMUX_STAT_BATT: No battery voltage detected
 * -D2 - HTSMUX_STAT_BUSY: Auto-dected in progress status
 * -D3 - HTSMUX_STAT_HALT: Multiplexer is halted
 * -D4 - HTSMUX_STAT_ERROR: Command error detected
 *
 * Note: this is an internal function and should not be called directly 
 * @param smux the SMUX port number
 * @return the status byte
 */
byte _HTSMUXreadStatus(const byte smux) {
  byte sendMsg[];
  byte readMsg[];
  const byte count = 1;
  
  ArrayBuild(sendMsg, HTSMUX_I2C_ADDR, HTSMUX_STATUS);

  // Query the SMUX and read the response
  while (I2CCheckStatus(smux) == STAT_COMM_PENDING) Yield();
  if (I2CBytes(smux, sendMsg, count, readMsg))
    return readMsg[0] & 0x00FF;
  else
    return 0;
}


/**
 * Scan the specified SMUX's channels and configure them. This function must be
 * called before the sensors attached to it can be queried.
 *
 * Note: this functions takes 500ms to return while the scan is in progress.
 * @param smux the SMUX port number
 * @return true if no error occured, false if it did
 */
bool HTSMUXscanPorts(const byte smux) {
  byte sendMsg[];
  byte readMsg[];

  byte foo = 0;  
  const byte size = 1;
  
  // Initialise array
  ArrayInit(sendMsg, 0, 2);
  
  // If we're in the middle of a scan, abort this call
  if (smuxStatus[smux] == HTSMUX_STAT_BUSY)
    return false;
	
  // Always make sure the SMUX is in the halted state
  if (!_HTSMUXsendCmd(smux, HTSMUX_CMD_HALT))
    return false;

  // Commence scanning the ports and allow up to 500ms to complete
  if (!_HTSMUXsendCmd(smux, HTSMUX_CMD_AUTODETECT))
    return false;
  Wait(500);
  
  // When the SMUX is done with scanning, it reverts back to halted mode.
  smuxStatus[smux] = HTSMUX_STAT_HALT;

  // Populate the smuxSensor array with the sensor types
  for (int i = 0; i < 4; i++) {
    // Query the SMUX and read the response
    while (I2CCheckStatus(smux) == STAT_COMM_PENDING) Yield();
    sendMsg[0] = HTSMUX_I2C_ADDR;   // I2C Address
    sendMsg[1] = HTSMUX_CH_OFFSET + HTSMUX_TYPE + (HTSMUX_CH_ENTRY_SIZE * i);
    
    if (I2CBytes(smux, sendMsg, size, readMsg)) {
      smuxSensor(smux,i) = readMsg[0];
    } else {
      smuxSensor(smux,i) = HTSMUXSensorNone;
    }
  }
  return true;
}


/**
 * Set the mode of a SMUX channel.
 *
 * Mode can be one or more of the following:
 * -HTSMUX_CHAN_I2C
 * -HTSMUX_CHAN_9V
 * -HTSMUX_CHAN_DIG0_HIGH
 * -HTSMUX_CHAN_DIG1_HIGH
 * -HTSMUX_CHAN_I2C_SLOW
 * @param muxport the SMUX sensor port number
 * @param mode the mode to set the channel to
 * @return true if no error occured, false if it did
 */
bool _HTSMUXsetMode(const byte muxport, const byte mode) {
  // Work-around for built-in macros that can't grok this.
  byte smux = SPORT(muxport);
  byte channel = MPORT(muxport);

  byte sendMsg[];
  byte regaddr = HTSMUX_CH_OFFSET + HTSMUX_MODE + (HTSMUX_CH_ENTRY_SIZE * channel);
  
  ArrayBuild(sendMsg, HTSMUX_I2C_ADDR, regaddr, mode);
  
  // If we're in the middle of a scan, abort this call
  if (smuxStatus[smux] == HTSMUX_STAT_BUSY) {
    return false;
  } else if (smuxStatus[smux] != HTSMUX_STAT_HALT) {
	  // Always make sure the SMUX is in the halted state
	  if (!_HTSMUXsendCmd(smux, HTSMUX_CMD_HALT))
	    return false;
	}

  // Send the command to the SMUX
  while (I2CCheckStatus(smux) == STAT_COMM_PENDING) Yield();
  I2CWrite(smux, 0, sendMsg);
  
  while ((I2CCheckStatus(smux) == STAT_COMM_PENDING) && (I2CCheckStatus(smux) != NO_ERR)) Yield();
  return (I2CCheckStatus(smux) == NO_ERR);
}


/**
 * Set the mode of an analogue channel to Active (turn the light on)
 * @param muxport the SMUX sensor port number
 * @return true if no error occured, false if it did
 */
bool HTSMUXsetAnalogueActive(const byte muxport) {
  if(smuxSensorType(muxport) != HTSMUXAnalogue)
    return false;

  if (!_HTSMUXsetMode(muxport, HTSMUX_CHAN_DIG0_HIGH))
    return false;

  return _HTSMUXsendCmd(SPORT(muxport), HTSMUX_CMD_RUN);
}


/**
 * Set the mode of an analogue channel to Inactive (turn the light off)
 * @param muxport The SMUX sensor port number
 * @return True if no error occured, false if it did
 */
bool HTSMUXsetAnalogueInactive(const byte muxport) {
  if(smuxSensorType(muxport) != HTSMUXAnalogue)
    return false;

  if (!_HTSMUXsetMode(muxport, 0))
    return false;

  return _HTSMUXsendCmd(SPORT(muxport), HTSMUX_CMD_RUN);
}


/**
 * Read LEGO Ultra Sound sensor.
 *
 * Read LEGO Ultra Sound sensor on the specified smux port.
 * @param muxport The SMUX sensor port number
 * @return The LEGO US sensor reading.
 */
int smuxSensorLegoUS(const byte muxport) {
  byte sensorVal[];
  
  if (!HTSMUXreadDigital (muxport, sensorVal, 0, 1))
    return 255;
  else
    return sensorVal[0] & 0x00FF;
}


/**
 * Read HiTechnic Gyro sensor.
 *
 * Read the HiTechnic Gyro sensor on the specified smux port. The offset value should
 * be calculated by averaging several readings with an offset of zero while the 
 * sensor is perfectly still.
 * @param muxport The SMUX sensor port number.
 * @param offset The zero offset.
 * @return The Gyro sensor reading.
 */
int smuxSensorHTGyro(const byte muxport, const int offset) {
  return HTSMUXreadAnalogue (muxport) - offset;
}


/** 
 * Set sensor as HiTechnic EOPD.
 *
 * Configure the sensor on the specified port as a HiTechnic EOPD sensor.
 * @param muxport The SMUX sensor port number.
 * @param bStandard Configure in standard or long-range mode.
 * @return True if no error occured, false if it did
 */
bool smuxSetSensorHTEOPD(const byte muxport, const bool bStandard) {
  // short range is set by making the sensor analogue inactive
  if (bStandard)
    return HTSMUXsetAnalogueInactive(muxport);
  else
    return HTSMUXsetAnalogueActive(muxport);
}


/**
 * Read HiTechnic EOPD sensor (raw value).
 *
 * Read the HiTechnic EOPD sensor (raw value) on the specified port.
 * @param muxport The SMUX sensor port number.
 * @return The EOPD sensor reading (raw value).
 */ 
int smuxSensorRawHTEOPD(const byte muxport) {
  return 1023 - HTSMUXreadAnalogue(muxport);
}


/**
 * Read HiTechnic EOPD sensor (processed value).
 *
 * Read the HiTechnic EOPD sensor (processed value) on the specified port.
 * @param muxport The SMUX sensor port number.
 * @return The EOPD sensor reading (processed value).
 */ 
int smuxSensorHTEOPD(const byte muxport) {
  return sqrt(smuxSensorRawHTEOPD(muxport) * 10);
}


/**
 * Read HiTechnic compass.
 *
 * Read the compass heading value of the HiTechnic Compass sensor on the 
 * specified port.
 * @param muxport The SMUX sensor port number.
 * @return The compass heading.
 */
int smuxSensorHTCompass(const byte muxport) {
  byte readMsg[];
  
  // Check if the attached sensor really -is- a compass sensor
  if (smuxSensorType(muxport) != HTSMUXCompass)
    return -1;
    
  if (!HTSMUXreadDigital (muxport, readMsg, 0, 2))
    return -1;
  else
    return ((readMsg[0]  & 0x00FF) * 2) + (readMsg[1] & 0x00FF);
}


/**
 * Read HiTechnic color sensor color number.
 *
 * Read the color number from the HiTechnic Color sensor on the specified port. 
 * @param muxport The SMUX sensor port number.
 * @return The color number.
 */
int smuxSensorHTColorNum(const byte muxport) {
  byte readMsg[];
  
  // Check if the attached sensor really -is- a colour sensor
  if ((smuxSensorType(muxport) != HTSMUXColor) 
   && (smuxSensorType(muxport) != HTSMUXColorNew))
    return -1;
      
  if (!HTSMUXreadDigital (muxport, readMsg, 0, 1))
    return 0;
  else
    return readMsg[0] & 0x00FF;
}


/**
 * Read HiTechnic Color values.
 *
 * Read the colour number, red, green, and blue values from the HiTechnic Color sensor.
 * Returns a boolean value indicating whether or not the operation completed 
 * successfully
 * @param muxport The SMUX sensor port number.
 * @param colnum The colour number
 * @param red The red color value.
 * @param green The green color value.
 * @param blue The raw blue color value.
 * @return True if no error occured, false if it did
 */
bool smuxReadSensorHTColor(const byte muxport, int &colnum, int &red, int &green, int &blue) {
  byte readMsg[];
  
  // Check if the attached sensor really -is- a colour sensor
  if ((smuxSensorType(muxport) != HTSMUXColor) 
   && (smuxSensorType(muxport) != HTSMUXColorNew))
    return false;
      
  if (!HTSMUXreadDigital (muxport, readMsg, 0, 4))
    return false;

  colnum = readMsg[0] & 0x00FF;
  red    = readMsg[1] & 0x00FF;
  green  = readMsg[2] & 0x00FF;
  blue   = readMsg[3] & 0x00FF;
  return true;
}


/**
 * Read HiTechnic IRSeeker direction.
 * 
 * Read the direction value of the HiTechnic IR Seeker on the specified port. 
 * @param muxport The sensor port.
 * @return The IRSeeker direction.
 */
int smuxSensorHTIRSeekerDir(const byte muxport) {
  byte readMsg[];

  // Check if the attached sensor really -is- a compass sensor
  if ((smuxSensorType(muxport) != HTSMUXIRSeeker) 
   && (smuxSensorType(muxport) != HTSMUXIRSeekerNew))
    return -1;
      
  if (!HTSMUXreadDigital (muxport, readMsg, 0, 1))
    return 0;
  else
    return readMsg[0] & 0x00FF;
}


/**
 * Read HiTechnic IRSeeker values.
 * 
 * Read direction, and five signal strength values from the HiTechnic IRSeeker 
 * sensor. Returns a boolean value indicating whether or not the operation 
 * completed successfully. 
 * @param muxport The sensor port.
 * @param dir The direction.
 * @param s1 The signal strength from sensor 1.
 * @param s2 The signal strength from sensor 2.
 * @param s3 The signal strength from sensor 3.
 * @param s4 The signal strength from sensor 4.
 * @param s5 The signal strength from sensor 5. 
 * @return True if no error occured, false if it did
 */
bool smuxReadSensorHTIRSeeker(const byte muxport, byte &dir, byte &s1, byte &s2, byte &s3, byte &s4, byte &s5) {
  byte readMsg[];
  
  // Check if the attached sensor really -is- a compass sensor
  if (smuxSensorType(muxport) != HTSMUXIRSeekerNew)
    return false;
      
  if (!HTSMUXreadDigital (muxport, readMsg, 0, 6))
    return false;

  dir = readMsg[0];
  s1  = readMsg[1];
  s2  = readMsg[2];
  s3  = readMsg[3];
  s4  = readMsg[4];
  s5  = readMsg[5];
  
  return true; 
}


/**
 * Read HiTechnic IRSeeker2 DC direction.
 * 
 * Read the DC direction value from the HiTechnic IR Seeker2 on the specified port.
 * @param muxport The sensor port.
 * @return The IRSeeker2 DC direction.
 */
int smuxSensorHTIRSeeker2DCDir(const byte muxport) {
  return smuxSensorHTIRSeekerDir(muxport);
}


/**
 * Read HiTechnic IRSeeker2 DC values.
 * 
 * Read direction, five signal strength, and average strength values from the 
 * HiTechnic IRSeeker2 sensor. Returns a boolean value indicating whether or not
 * the operation completed successfully 
 * @param muxport The sensor port.
 * @param dir The direction.
 * @param s1 The signal strength from sensor 1.
 * @param s2 The signal strength from sensor 2.
 * @param s3 The signal strength from sensor 3.
 * @param s4 The signal strength from sensor 4.
 * @param s5 The signal strength from sensor 5. 
 * @param avg The average signal strength.
 * @return True if no error occured, false if it did
 */
bool smuxReadSensorHTIRSeeker2DC(const byte muxport, byte &dir, byte &s1, byte &s2, byte &s3, byte &s4, byte &s5, byte &avg) {
  byte readMsg[];
  
  // Check if the attached sensor really -is- a compass sensor
  if (smuxSensorType(muxport) != HTSMUXIRSeekerNew)
    return false;
      
  if (!HTSMUXreadDigital (muxport, readMsg, 0, 7))
    return false;

  dir = readMsg[0];
  s1  = readMsg[1];
  s2  = readMsg[2];
  s3  = readMsg[3];
  s4  = readMsg[4];
  s5  = readMsg[5];
  avg = readMsg[6];
  
  return true; 
}


/**
 * Read HiTechnic IRSeeker2 AC direction.
 * 
 * Read the AC direction value from the HiTechnic IR Seeker2 on the specified port.
 * @param muxport The sensor port.
 * @return The IRSeeker2 AC direction.
 */
int smuxSensorHTIRSeeker2ACDir(const byte muxport) {
  byte readMsg[];
  
  // Check if the attached sensor really -is- a compass sensor
  if (smuxSensorType(muxport) != HTSMUXIRSeekerNew)
    return -1;
      
  if (!HTSMUXreadDigital (muxport, readMsg, 7, 1))
    return 0;
  else
    return readMsg[0] & 0x00FF;
}


/**
 * Read HiTechnic IRSeeker2 AC values.
 * 
 * Read direction, and five signal strength values from the HiTechnic IRSeeker2 
 * sensor in AC mode. Returns a boolean value indicating whether or not
 * the operation completed successfully 
 * @param muxport The sensor port.
 * @param dir The direction.
 * @param s1 The signal strength from sensor 1.
 * @param s2 The signal strength from sensor 2.
 * @param s3 The signal strength from sensor 3.
 * @param s4 The signal strength from sensor 4.
 * @param s5 The signal strength from sensor 5. 
 * @return True if no error occured, false if it did
 */
bool smuxReadSensorHTIRSeeker2AC(const byte muxport, byte &dir, byte &s1, byte &s2, byte &s3, byte &s4, byte &s5) {
  byte readMsg[];
  
  // Check if the attached sensor really -is- a compass sensor
  if (smuxSensorType(muxport) != HTSMUXIRSeekerNew)
    return false;
      
  if (!HTSMUXreadDigital (muxport, readMsg, 7, 6))
    return false;

  dir = readMsg[0];
  s1  = readMsg[1];
  s2  = readMsg[2];
  s3  = readMsg[3];
  s4  = readMsg[4];
  s5  = readMsg[5];
  
  return true;  
}
	

/**
 * Read HiTechnic acceleration values.
 * 
 * Read X, Y, and Z axis acceleration values from the HiTechnic Accelerometer 
 * sensor. Returns a boolean value indicating whether or not the operation 
 * completed successfully. 
 * @param muxport The sensor port.
 * @param x The output x-axis acceleration.
 * @param y The output y-axis acceleration.
 * @param z The output z-axis acceleration.
 * @return True if no error occured, false if it did
 */
bool smuxReadSensorHTAccel(const byte muxport, int &x, int &y, int &z) {
  byte readMsg[];
  
  if (smuxSensorType(muxport) != HTSMUXAccel)
    return false;

  if (!HTSMUXreadDigital(muxport, readMsg, 0, 6)) {
    return false;
  }

  // Convert 2 bytes into a signed 10 bit value.  If the 8 high bits are more than 127, make
  // it a signed value before combing it with the lower 2 bits.
  // Gotta love conditional assignments!
  x = (readMsg[0] > 127) ? ((readMsg[0] - 256) * 4) + readMsg[3]
                                   : (readMsg[0] * 4) + readMsg[3];

  y = (readMsg[1] > 127) ? ((readMsg[1] - 256) * 4) + readMsg[4]
                                   : (readMsg[1] * 4) + readMsg[4];

  z = (readMsg[2] > 127) ? ((readMsg[2] - 256) * 4) + readMsg[5]
                                   : (readMsg[2] * 4) + readMsg[5];
  
  return true;
}

#endif // __HTSMUX_DRIVER_H__


/**
 * Read LEGO Light sensor (raw value).
 *
 * Read LEGO Lightsensor (raw value) on the specified port.
 * @param muxport The SMUX sensor port number.
 * @return Read LEGO Light sensor reading (raw value).
 */ 
int smuxSensorLegoLightRaw(const byte muxport) {
  return 1023 - HTSMUXreadAnalogue(muxport);
}


/**
 * Read LEGO Light sensor (normalised value).
 *
 * Read LEGO Lightsensor (normalised value) on the specified port.
 * @param muxport The SMUX sensor port number.
 * @return Read LEGO Light sensor reading (normalised value).
 */ 
int smuxSensorLegoLightNorm(const byte muxport) {
  long val = smuxSensorLegoLightRaw(muxport);
  return (val * 100) / 1024;
}


/** 
 * Set sensor as LEGO Light sensor.
 *
 * Configure the sensor on the specified port as a LEGO Light sensor.
 * @param muxport The SMUX sensor port number.
 * @param active Configure in active or inactive mode.
 * @return True if no error occured, false if it did
 */
bool smuxSetSensorLegoLight(const byte muxport, const bool active) {
  if (!active)
    return HTSMUXsetAnalogueInactive(muxport);
  else
    return HTSMUXsetAnalogueActive(muxport);
}


/**
 * Read HiTechnic EOPD sensor (raw value).
 *
 * Read the HiTechnic EOPD sensor (raw value) on the specified port.
 * @param muxport The SMUX sensor port number.
 * @return The EOPD sensor reading (raw value).
 */ 
int smuxSensorLegoSoundRaw(const byte muxport) {
  return 1023 - HTSMUXreadAnalogue(muxport);
}


/**
 * Read LEGO Sound sensor (normalised value).
 *
 * Read LEGO Sound sensor (normalised value) on the specified port.
 * @param muxport The SMUX sensor port number.
 * @return Read LEGO Sound sensor reading (normalised value).
 */ 
int smuxSensorLegoSoundNorm(const byte muxport) {
  long val = smuxSensorLegoSoundRaw(muxport);
  return (val * 100) / 1024;
}


/** 
 * Set sensor as LEGO Sound sensor.
 *
 * Configure the sensor on the specified port as a LEGO Sound sensor.
 * @param muxport The SMUX sensor port number.
 * @param dba Configure in dBA or dB mode.
 * @return True if no error occured, false if it did
 */
bool smuxSetSensorLegoSound(const byte muxport, const bool dba) {
  if (dba)
    return HTSMUXsetAnalogueInactive(muxport);
  else
    return HTSMUXsetAnalogueActive(muxport);
}


/**
 * Read LEGO Touch sensor
 *
 * Read LEGO Touch sensor on the specified port.
 * @param muxport The SMUX sensor port number.
 * @return Read LEGO Touch sensor value.
 */
bool smuxReadSensorLegoTouch(const byte muxport)
{
  if ( HTSMUXreadAnalogue(muxport) < TOUCH_SENSOR_THRESHOLD )
    return TRUE;
  else
    return FALSE;
}


/**
 * Set sensor as LEGO Sound sensor.
 *
 * Configure the sensor on the specified port as a LEGO Touch sensor.
 * @param muxport The SMUX sensor port number.
 * @return True if no error occured, false if it did
 */
bool smuxSetSensorLegoTouch(const byte muxport) {
  return HTSMUXsetAnalogueInactive(muxport);
}


/*
 * $Id$
 */
 
