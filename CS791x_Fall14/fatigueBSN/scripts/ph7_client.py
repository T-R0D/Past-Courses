from bluetooth import *
import sys

addr = "00:22:D0:3E:4B:04"
uuid = "0000180d-0000-1000-8000-00805f9b34fb"

nbd = discover_devices(lookup_names=True)

for addr, name in nbd:
  print "Address: %s Name: %s" % (addr, name)
service_matches = find_service(uuid =uuid, address=addr)


if len(service_matches) <= 0:
    print("No device found")
    sys.exit(0)

first_match = service_matches[0]
port = first_match["port"]
name = first_match["name"]
host = first_match["host"]

print("connecting to \"%s\" on %s" % (name, host))
dummy = raw_input()

# Create the client socket
sock = BluetoothSocket(RFCOMM)
sock.connect((host, port))

print("connected.  type stuff")
while True:
    data = raw_input()
    if len(data) == 0:
        break
    sock.send(data)

sock.close()