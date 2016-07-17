import bluetooth
import time


class h7PolarRawReader:
    START_OF_PACKET_BYTE = 0xfe;
    def __init__(self):
        self._buffer = [];
        self._bufferPosition = 0;
        
    def connectToh7Polar(self):
        # connecting via bluetooth RFCOMM
        self.h7PolarSocket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        h7PolarAddress = '00:22:D0:3E:4B:04';
        while(True):
            try:
                self.h7PolarSocket.connect((h7PolarAddress, 1))
                return;
            except bluetooth.btcommon.BluetoothError as error:
                print "Could not connect: ", error, "; Retrying in 5s..."
                time.sleep(5) 
    
    def _readMoreBytesIntoBuffer(self, amountOfBytes):
        newBytes = self._readBytesFromh7Polar(amountOfBytes)
        self._buffer += newBytes
    
    def _readBytesFromh7Polar(self, amountOfBytes):
        missingBytes = amountOfBytes
        receivedBytes = ""
        # Sometimes the socket will not send all the requested bytes
        # on the first request, therefore a loop is necessary...
        while(missingBytes > 0):
            receivedBytes += self.h7PolarSocket.recv(missingBytes)
            missingBytes = amountOfBytes - len(receivedBytes)
        return receivedBytes;

    def peekByte(self):
        self._ensureMoreBytesCanBeRead();
        return ord(self._buffer[self._bufferPosition])

    def getByte(self):
        self._ensureMoreBytesCanBeRead(100);
        return self._getNextByte();
    
    def  _ensureMoreBytesCanBeRead(self, amountOfBytes):
        if (self._bufferSize() <= self._bufferPosition + amountOfBytes):
            self._readMoreBytesIntoBuffer(amountOfBytes)
    
    def _getNextByte(self):
        nextByte = ord(self._buffer[self._bufferPosition]);
        self._bufferPosition += 1;
        return nextByte;

    def getBytes(self, amountOfBytes):
        self._ensureMoreBytesCanBeRead(amountOfBytes);
        return self._getNextBytes(amountOfBytes);
    
    def _getNextBytes(self, amountOfBytes):
        nextBytes = map(ord, self._buffer[self._bufferPosition: self._bufferPosition + amountOfBytes])
        self._bufferPosition += amountOfBytes
        return nextBytes
    
    def clearAlreadyReadBuffer(self):
        self._buffer = self._buffer[self._bufferPosition : ]
        self._bufferPosition = 0;
    
    def _bufferSize(self):
        return len(self._buffer);
    
#------------------------------------------------------------------------------ 
