#!/usr/bin/env python

import time, struct,sys
import bluetooth
from mindwavemobile.MindwaveDataPoints import AttentionDataPoint, EEGPowersDataPoint
from mindwavemobile.MindwaveDataPointReader import MindwaveDataPointReader

import numpy as np
import pylab as pl

def main():
  mdpr = MindwaveDataPointReader()
  mdpr.start()
  eeg_datapoints = []
  attention_datapoints = []

  index = 0
  try:
    while(True):
      data = mdpr.readNextDataPoint()
      if (data.__class__ is AttentionDataPoint):
        attention_datapoints.append((time.time(),data))
      if (data.__class__ is EEGPowersDataPoint):
        eeg_datapoints.append((time.time(),data))
        index+=1
        print index
  except KeyboardInterrupt:
    pass
  fmt = 'ddddddddd'
  dataFormat = []
  file_ = open(sys.argv[1], 'wb')
  file_.write(fmt.ljust(25,' '))
  for i in xrange(len(eeg_datapoints)):
    timestamp = attention_datapoints[i][0]
    attention = attention_datapoints[i][1]
    delta     = eeg_datapoints[i][1].delta
    theta     = eeg_datapoints[i][1].theta
    lowalpha  = eeg_datapoints[i][1].lowAlpha
    highalpha = eeg_datapoints[i][1].highAlpha
    lowbeta   = eeg_datapoints[i][1].lowBeta
    highbeta  = eeg_datapoints[i][1].highBeta
    lowgamma  = eeg_datapoints[i][1].lowGamma
    midgamma = eeg_datapoints[i][1].midGamma


    s = struct.pack(fmt,timestamp, delta, theta, lowalpha, highalpha, lowbeta, highbeta, lowgamma, midgamma)
    file_.write(s)
  file_.close()

if __name__ == '__main__':
  main()