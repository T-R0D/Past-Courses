import socket
import sys


class AndroidImuDataPoint(object):
    def __init__(self, dataFromAppAsString):
        data = dataFromAppAsString.split(",")
        
        self.sampleTime = data[0]

        # each item is a list of 3 items in the order X, Y, Z
        i = 1
        self.accelerometer = [float(x) for x in data[i:i + 3]]

        i += 3
        self.gyroscope = [float(x) for x in data[i:i + 3]]

        i += 3
        self.magnet = [float(x) for x in data[i:i + 3]]

        i += 3
        self.orientation = [float(x) for x in data[i:i + 3]]
        # X = roll of phone, Y = pitch of phone, Z = yaw of phone

    def __str__(self):
        return """
    Time:        {}
                   X      Y     Z
    ORIENTATION: {}
    Accelration: {}
    Gyroscope:   {}""".format(
        self.sampleTime,
        self.accelerometer,
        self.gyroscope,
        self.orientation
    )
            

host = ''
port = 5555

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
s.bind((host, port))

#used for debugging

print("Success binding")
with open(sys.argv[1], "w") as data_file:
    try:
        while True:
            message, address = s.recvfrom(8192)
            messageString = message.decode("utf-8")
            dataPoint = AndroidImuDataPoint(messageString)
            data_file.write(messageString + '\n')
            print(dataPoint)
    except KeyboardInterrupt:
        pass
    data_file.close()

