#!/usr/bin/env python

from __future__ import print_function
import pylab as pl
import merge_sensor_data as msd

LUKE_FILE = '/home/t/Desktop/fatigueBSN/fatigue_test_data/Luke/12_03_2202_heart.dat'
TERENCE_FILE = '/home/t/Desktop/fatigueBSN/fatigue_test_data/Terence/12_03_2002_cleaned_imu.dat'

def main():
	data, fmt = msd.unpack_binary_data_into_list(TERENCE_FILE)

	# data format:
	# time, a_x, a_y, a_z, g_x, g_y, g_z
	g_x = [x[4] for x in data]
	g_y = [x[5] for x in data]
	g_z = [x[6] for x in data]
	sums = []
	for i in range(len(g_z)):
		sums.append(g_x[i] + g_y[i] + g_z[i])

	pl.plot(range(len(g_x)), g_x, label='Gyroscope_x')
	pl.plot(range(len(g_y)), g_y, label='Gyroscope_y')
	pl.plot(range(len(g_z)), g_z, label='Gyroscope_z')
	# pl.plot(range(len(g_z)), sums, label='sums')
	pl.xlabel("Time Step")
	pl.ylabel("Gyroscope Reading (Radians)")
	pl.title("IMU Visualization")
	legend = pl.legend(loc='best', ncol=2, shadow=None)
	legend.get_frame().set_facecolor('#00FFCC')
	pl.show()


if __name__ == '__main__':
	main()