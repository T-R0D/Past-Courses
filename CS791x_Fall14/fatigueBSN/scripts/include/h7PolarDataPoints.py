   
class DataPoint:
    def __init__(self, dataValueBytes):
        self._dataValueBytes = dataValueBytes

class h7PolarDataPoint(DataPoint):
	def __init__(self, dataValueBytes):
		DataPoint.__init__(self, dataValueBytes)
		self.status = self.dataValueBytes[4]
		self.heartRate = dataValueBytes[5]
                
	def __str__(self):
		return "Heart Rate: " + self.heartRate

