#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <bluetooth/bluetooth.h>
#include <bluetooth/hci.h>
#include <bluetooth/hci_lib.h>

#include "ThinkGearStreamParser.h"


int other_main(int argc, char **argv);
void handleDataValueFunc( unsigned char extendedCodeLevel,
                          unsigned char code,
                          unsigned char valueLength,
                          const unsigned char *value,
                          void *customData );


/**
 * Program which reads ThinkGear Data Values from a COM port.
 */
int main(int argc, char **argv)
{

    /* 2) Initialize ThinkGear stream parser */
    ThinkGearStreamParser parser;
    THINKGEAR_initParser( &parser,             // the parser object, tracks the state of things
                          PARSER_TYPE_PACKETS, //
                          handleDataValueFunc, // our callback function to handle data
                          NULL                 // we aren't passing any data to the parser
    );
 
    /* TODO: Initialize 'stream' here to read from a serial data
     * stream, or whatever stream source is appropriate for your
     * application.  See documentation for "Serial I/O" for your
     * platform for details.
     */
    struct sockaddr_rc loc_addr, rem_addr; //= { 0, 0, 0 }, rem_addr = {0, 0, 0 };
    char buf[1024] = { 0 };
    int s, client, bytes_read;
    socklen_t opt = sizeof(rem_addr);

    // allocate socket
    s = socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM);

    // bind socket to port 1 of the first available 
    // local bluetooth adapter
    loc_addr.rc_family = AF_BLUETOOTH;
    loc_addr.rc_bdaddr = *BDADDR_ANY;
    loc_addr.rc_channel = (uint8_t) 1;
    bind(s, (struct sockaddr *)&loc_addr, sizeof(loc_addr));

    // put socket into listening mode
    listen(s, 1);

    // accept one connection
    client = accept(s, (struct sockaddr *)&rem_addr, &opt);

    ba2str( &rem_addr.rc_bdaddr, buf );
    fprintf(stderr, "accepted connection from %s\n", buf);
    memset(buf, 0, sizeof(buf));



    /* 3) Stuff each byte from the stream into the parser.  Every time
     *    a Data Value is received, handleDataValueFunc() is called.
     */
    unsigned char streamByte;
    while(1) {
      // read data from the client
      bytes_read = read(client, buf, sizeof(buf));
      if( bytes_read > 0 ) {
          printf("received [%s]\n", buf);
      }

      THINKGEAR_parseByte( // this will call our callback function once parsing is finished
          &parser,
          bytes_read
      ); 
    }

    // close connection
    close(client);
    close(s);
    return 0;





  // A simplescan.c but reworded  
  // inquiry_info *inquiryInfo = NULL; // to be an array of device info structs
  // int max_rsp,  // maximum number of Blutooth devices to query for
  //     num_rsp;  // number of Bluetooth devices found
  // int dev_id,   // id of the Bluetooth adapter
  //     socket,   // the socket corresponding to the Bluetooth adapter
  //     len,      // an inquiry wait time multiplier
  //     flags;    // a flag to indicate using cached device info or clear it
  // int i;
  // char addr[19] = { 0 }; // c-string for a Bluetooth address
  // char name[248] = { 0 }; // c-string for Bluetooth device English name

  // dev_id = hci_get_route(NULL); // [NULL] gets the resource number of the first
  //                               // available Bluetooth adapter
  // // we could alternately use
  // //    deviceId = hci_devid("01:23:45:67:89:AB");
  // // for example, if we knew a specific Bluetooth adapter address

  // socket = hci_open_dev(dev_id); // opens a bluetooth socket with the given
  //                                // resource number

  // if (dev_id < 0 || socket < 0) { // if either operation failed...
  //     perror("opening socket");
  //     exit(1);
  // }

  // len  = 8;
  // max_rsp = 255;
  // flags = IREQ_CACHE_FLUSH; // this flag says to flush the cache of remembered
  //                           // devices; using 0 would retain them
  // inquiryInfo = (inquiry_info*) malloc(max_rsp * sizeof(inquiry_info));
  
  // // scan for Bluetooth devices
  // num_rsp = hci_inquiry(
  //   dev_id, // the resource number of the adapter to check on
  //   len,    // 1.28 * len seconds: the maximum inquiry time
  //   max_rsp, // the maximum number of devices to return (in inquiryInfo)
  //   NULL,    // ???
  //   &inquiryInfo, // the list (array) of detected devices - returned
  //   flags    // whether or not to use cached devices
  // );

  // if( num_rsp < 0 ) { // no devices found (-1 was returned on error)
  //   perror("hci_inquiry");
  // }

  // for (i = 0; i < num_rsp; i++) { // for all found devices
  //   ba2str(&(inquiryInfo+i)->bdaddr, addr); // convert their Bluetooth address
  //                                           // to a string

  //   memset(name, 0, sizeof(name)); // zero (null) out our name c-string

  //   int remoteReadResult = hci_read_remote_name( // query the device name
  //     socket,            // the socket being used for the Bluetooth adapter
  //     &(inquiryInfo+i)->bdaddr, // the Bluetooth address of the device
  //     sizeof(name),             // how many bytes of the name to copy
  //     name,                     // where to store the copied bytes
  //     0                         // timeout (ms) for using the socket
  //   );
  //   if ( remoteReadResult < 0) { // 0 == success; -1 == failure
  //     strcpy(name, "[unknown]");
  //   }
  //   printf("%s  %s\n", addr, name);
  // }

  // free(inquiryInfo); // release dynamic memory
  // close(socket);     // close the socket
  // return 0;
}


/**
 * 1) Function which acts on the value[] bytes of each ThinkGear DataRow as it is received.
 */
void handleDataValueFunc( unsigned char extendedCodeLevel,
                          unsigned char code,
                          unsigned char valueLength,
                          const unsigned char *value,
                          void *customData ) {
 
    if (extendedCodeLevel == 0) {
        switch (code) {
            case( 0x04 ): /* [CODE]: ATTENTION eSense */
                printf( "Attention Level: %d\n", value[0] & 0xFF );
                break;
            case( 0x05 ): /* [CODE]: MEDITATION eSense */
                printf( "Meditation Level: %d\n", value[0] & 0xFF );
                break;
            default: /* Other [CODE]s */
                printf( "EXCODE level: %d CODE: 0x%02X vLength: %d\n",
                        extendedCodeLevel, code, valueLength );
                printf( "Data value(s):" );
                for(int i=0; i<valueLength; i++ ) printf( " %02X", value[i] & 0xFF );
                printf( "\n" );
        }
    }
}