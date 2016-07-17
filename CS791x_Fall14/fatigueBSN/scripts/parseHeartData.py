#!/usr/bin/env python

import re, sys, struct

def main():
  heart_file = open(sys.argv[1], 'r')
  heart_val = []
  timestamp = []
  for line in heart_file:
    if line[:2] == "14":
      timestamp.append(int(line[:-1])/1000000000.0)
    if line[0] == 'N' and (line[36:38] == "16"):
      heart_val.append(int(line[39:41],16))

  duration = timestamp[1]- timestamp[0]
  delta = duration / len(heart_val)
  for i in xrange(len(heart_val)):
    heart_val[i] = [timestamp[0] + delta*i, heart_val[i]]

  heart_file.close()

  output = open(sys.argv[2], 'wb')
  fmt = 'di'
  output.write(fmt.ljust(25,' '))
  for val in heart_val:
    output.write(struct.pack(fmt, *val))
  output.close()
if __name__ == '__main__':
  main()