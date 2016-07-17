from h7PolarRawReader import h7PolarRawReader
import struct
import collections

from h7PolarPacketPayloadParser import h7PolarPacketPayloadParser

class h7PolarDataPointReader:
    def __init__(self):
        self._h7PolarRawReader = h7PolarRawReader()
        self._dataPointQueue = collections.deque()

    def start(self):
	print ("attempting to connect")
        self._h7PolarRawReader.connectToh7Polar()
	print ("finished attempting to connect")
        
    def readNextDataPoint(self):
        if (not self._moreDataPointsInQueue()):
            self._putNextDataPointsInQueue()
        return self._getDataPointFromQueue()

    def _moreDataPointsInQueue(self):
        return len(self._dataPointQueue) > 0
    
    def _getDataPointFromQueue(self):
        return self._dataPointQueue.pop();
    
    def _putNextDataPointsInQueue(self):
        dataPoints = self._readDataPointsFromOnePacket()
        self._dataPointQueue.extend(dataPoints)
    
    def _readDataPointsFromOnePacket(self):
        self._goToStartOfNextPacket()
        payloadBytes, checkSum = self._readOnePacket()
        if (not self._checkSumIsOk(payloadBytes, checkSum)):
            print "checksum of packet was not correct, discarding packet..."
            return self._readDataPointsFromOnePacket();
        else:
            dataPoints = self._readDataPointsFromPayload(payloadBytes)
        self._h7PolarRawReader.clearAlreadyReadBuffer()
        return dataPoints;
        
    def _goToStartOfNextPacket(self):
        while(True):
            byte = self._h7PolarRawReader.getByte()
            if (byte == h7PolarRawReader.START_OF_PACKET_BYTE):  # need two of these bytes at the start..
                # now at the start of the packet..
                return;

    def _readOnePacket(self):
            payloadLength = self._readPayloadLength();
            payloadBytes, checkSum = self._readPacket(payloadLength);
            return payloadBytes, checkSum
    
    def _readPayloadLength(self):
        payloadLength = self._h7PolarRawReader.getByte()
        return payloadLength

    def _readPacket(self, payloadLength):
        payloadBytes = self._h7PolarRawReader.getBytes(payloadLength)
        checkSum = self._h7PolarRawReader.getByte()
        return payloadBytes, checkSum

    def _checkSumIsOk(self, payloadBytes, checkSum):
        sumOfPayload = sum(payloadBytes)
        lastEightBits = sumOfPayload % 256
        invertedLastEightBits = self._computeOnesComplement(lastEightBits) #1's complement!
        return invertedLastEightBits == checkSum;
    
    def _computeOnesComplement(self, lastEightBits):
        return ~lastEightBits + 256
        
    def _readDataPointsFromPayload(self, payloadBytes):
        payloadParser = h7PolarPacketPayloadParser(payloadBytes)
        return payloadParser.parseDataPoints();
    
    
    
    
