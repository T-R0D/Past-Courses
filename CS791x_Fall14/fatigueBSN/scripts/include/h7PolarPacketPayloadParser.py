from h7PolarDataPoints import h7PolarDataPoint

EXTENDED_CODE_BYTE = 0x55

class h7PolarPacketPayloadParser:
    
    def __init__(self, payloadBytes):
        self._payloadBytes = payloadBytes
        self._payloadIndex = 0
        
    def parseDataPoints(self):
        dataPoints = []
        # while (not self._atEndOfPayloadBytes()):
        #     dataPoint = self._parseOneDataPoint()
        #     dataPoints.append(dataPoint)
        dataPoint = h7PolarDataPoint(self.payloadBytes)
        dataPoints.append(dataPoint)
        return dataPoints
        
    def _atEndOfPayloadBytes(self):
        return self._payloadIndex == len(self._payloadBytes)
    
    def _parseOneDataPoint(self):
        # dataRowCode = self._extractDataRowCode();
        dataRowValueBytes = self._extractDataRowValueBytes(dataRowCode)
        return self._createDataPoint(dataRowCode, dataRowValueBytes)
    
    def _extractDataRowCode(self):
        return self._ignoreExtendedCodeBytesAndGetRowCode()
        
    def _ignoreExtendedCodeBytesAndGetRowCode(self):
        # EXTENDED_CODE_BYTES seem not to be used according to 
        # http://wearcam.org/ece516/mindset_communications_protocol.pdf
        # (August 2012)
        # so we ignore them
        byte = self._getNextByte()
        while (byte == EXTENDED_CODE_BYTE):
            byte = self._getNextByte()
        dataRowCode = byte
        return dataRowCode
       
    def _getNextByte(self):
        nextByte = self._payloadBytes[self._payloadIndex]
        self._payloadIndex += 1
        return nextByte
    
    def _getNextBytes(self, amountOfBytes):
        nextBytes = self._payloadBytes[self._payloadIndex : self._payloadIndex + amountOfBytes]
        self._payloadIndex += amountOfBytes
        return nextBytes
    
    def _extractDataRowValueBytes(self, dataRowCode):
        lengthOfValueBytes = self._extractLengthOfValueBytes(dataRowCode)
        dataRowValueBytes = self._getNextBytes(lengthOfValueBytes)
        return dataRowValueBytes
       
    def _extractLengthOfValueBytes(self, dataRowCode):
        dataRowHasLengthByte = dataRowCode > 0x7f
        if (dataRowHasLengthByte):
            return self._getNextByte()
        else:
            return 1
        
    def _createDataPoint(self, dataRowCode, dataRowValueBytes):
        if (dataRowCode == 0x02):
            return h7PolarDataPoint(dataRowValueBytes)
        else:
            assert False 
        