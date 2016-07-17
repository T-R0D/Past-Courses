#!/usr/bin/expect

spawn "./heartmonitor.sh"

send "connect\n"
# send "char-write-req 0x0013 0100\n"
interact