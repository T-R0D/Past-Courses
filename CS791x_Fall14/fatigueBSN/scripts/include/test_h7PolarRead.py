
import time
import bluetooth
from h7PolarDataPoints import h7PolarDataPoint
from h7PolarDataPointReader import h7PolarDataPointReader


if __name__ == '__main__':
    h7PolarDataPointReader = h7PolarDataPointReader()
    h7PolarDataPointReader.start()
    
    while(True):
        dataPoint = h7PolarDataPointReader.readNextDataPoint()
        print (dataPoint)