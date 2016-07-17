#!/usr/bin/env python

import struct, sys, re, time, calendar

def main():
  imu_file = open(sys.argv[1], 'r')
  for i in xrange(7):
    imu_file.readline()

  t = time.gmtime()
  # print time.time()
  # print datTime
  imu_data = []
  for line in imu_file:
    imu_data.append(re.split('\t',line[:-1]))
  for i, step in enumerate(imu_data):
    if i != 0:
      timestamp = step[0]
      hour = int(timestamp[:2])
      minute = int(timestamp[3:5])
      sec = float(timestamp[6:])
      datTime = calendar.timegm(time.struct_time((t.tm_year, t.tm_mon, t.tm_mday, hour, minute, sec, t.tm_wday, t.tm_yday, t.tm_isdst)))
      imu_data[i][0] = datTime
  for i, step in enumerate(imu_data):
    for j, val in enumerate(step):
      if i != 0:
        imu_data[i][j] = float(val)

  f = open(sys.argv[2], 'wb')
  fmt = 'dddddddddddddddd'
  f.write(fmt.ljust(25,' '))

  for step in imu_data[1:]:
    f.write(struct.pack(fmt, *step))
  f.close()


if __name__ == '__main__':
  main()
