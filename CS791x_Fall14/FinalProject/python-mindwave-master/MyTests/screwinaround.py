import os
import sys
import bluetooth
from bluetooth.btcommon import BluetoothError
import json
import time
import struct
from datetime import datetime
import argparse

class ThinkGearParser(object):
    def __init__(self, recorders=None):
        self.recorders = []
        if recorders is not None:
            self.recorders += recorders
        self.input_data = ""
        self.parser = self.parse()
        self.parser.next()

    def feed(self, data):
        for c in data:
            self.parser.send(ord(c))
        for recorder in self.recorders:
            recorder.finish_chunk()
        self.input_data += data

    def dispatch_data(self, key, value):
        print(key + ": " + value)
        for recorder in self.recorders:
            recorder.dispatch_data(key, value)

    def parse(self):
        """
            This generator parses one byte at a time.
        """
        i = 1
        times = []
        while 1:
            byte = yield
            if byte== 0xaa:
                byte = yield # This byte should be "\aa" too
                if byte== 0xaa:
                    # packet synced by 0xaa 0xaa
                    packet_length = yield
                    packet_code = yield
                    if packet_code == 0xd4:
                        # standing by
                        self.state = "standby"
                    elif packet_code == 0xd0:
                        self.state = "connected"
                    elif packet_code == 0xd2:
                        data_len = yield
                        headset_id = yield
                        headset_id += yield
                        self.dongle_state = "disconnected"
                    else:
                        self.sending_data = True
                        left = packet_length - 2
                        while left>0:
                            if packet_code ==0x80: # raw value
                                row_length = yield
                                a = yield
                                b = yield
                                value = struct.unpack("<h",chr(b)+chr(a))[0]
                                self.dispatch_data("raw", value)
                                left -= 2
                            elif packet_code == 0x02: # Poor signal
                                a = yield

                                left -= 1
                            elif packet_code == 0x04: # Attention (eSense)
                                a = yield
                                if a>0:
                                    v = struct.unpack("b",chr(a))[0]
                                    if 0 < v <= 100:
                                        self.dispatch_data("attention", v)
                                left-=1
                            elif packet_code == 0x05: # Meditation (eSense)
                                a = yield
                                if a>0:
                                    v = struct.unpack("b",chr(a))[0]
                                    if 0 < v <= 100:
                                        self.dispatch_data("meditation", v)
                                left-=1
                            elif packet_code == 0x16: # Blink Strength
                                self.current_blink_strength = yield
                              
                                left-=1
                            elif packet_code == 0x83:
                                vlength = yield
                                self.current_vector = []
                                for row in range(8):
                                    a = yield
                                    b = yield
                                    c = yield
                                    value = a*255*255+b*255+c
                                left -= vlength
                                self.dispatch_data("bands", self.current_vector)
                            packet_code = yield
                else:
                    pass # sync failed
            else:
                pass # sync failed


def connect_bluetooth_addr(addr):
    for i in range(5):
        if i > 0:
            time.sleep(1)
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        try:
            sock.connect((addr, 1))
            sock.setblocking(False)
            return sock
        except BluetoothError, e:
            print e
    return None


def connect_magic():
    """ Tries to connect to the first MindWave Mobile it can find.
        If this computer hasn't connected to the headset before, you may need
        to make it visible to this computer's bluetooth adapter. This is done
        by pushing the switch on the left side of the headset to the "on/pair"
        position until the blinking rythm switches.

        The address is then put in a file for later reference.

    """
    nearby_devices = bluetooth.discover_devices(lookup_names = True, duration=5)

    for addr, name in nearby_devices:
        print(name)
        if name == "MindWave Mobile":
            print "found"
            return (connect_bluetooth_addr(addr), addr)
    return (None, "")


def mindwave_startup(description="", extra_args=[]):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('address', type=str, nargs='?',
            const=None, default=None,
            help="""Bluetooth Address of device. Use this
            if you have multiple headsets nearby or you want
            to save a few seconds during startup.""")
    for params in extra_args:
        name = params['name']
        del params['name']
        parser.add_argument(name, **params)
    args = parser.parse_args(sys.argv[1:])
    if args.address is None:
        socket, socket_addr = connect_magic()
        if socket is None:
            print "No MindWave Mobile found."
            sys.exit(-1)
    else:
        socket = connect_bluetooth_addr(args.address)
        if socket is None:
            print "Connection failed."
            sys.exit(-1)
        socket_addr = args.address
    print "Connected with MindWave Mobile at %s" % socket_addr
    for i in range(5):
        try:
            if i>0:
                print "Retrying..."
            time.sleep(2)
            len(socket.recv(10))
            break
        except BluetoothError, e:
            print e
        if i == 5:
            print "Connection failed."
            sys.exit(-1)
    return socket, args

#=======================================================================

print("Let's give this a try...")

socket_, args = mindwave_startup("20:68:9D:A8:4B:A8");
thinkGearParser = ThinkGearParser()


while true:
    time.sleep(0.25)
    data = socket.recv(20000)
    thinkGearParser.feed(data)





