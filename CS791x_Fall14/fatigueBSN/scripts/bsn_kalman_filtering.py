#!/usr/bin/env python

from __future__ import print_function
from merge_sensor_data import unpack_binary_data_into_list
import numpy as np
from pykalman import KalmanFilter
import struct
import string


BSN_DATA_FILE_NAME =\
	"/home/t/Desktop/fatigueBSN/fatigue_test_data/Terence/12_03_2002_merged.dat"
FILTERED_BSN_DATA_FILE_NAME = "f/home/t/Desktop/fatigueBSN/fatigue_test_data/Terence/12_03_2002_filtered_merged.dat"

def main():
	# get the data
	readings, data_format = unpack_binary_data_into_list(BSN_DATA_FILE_NAME)
	# just the data/readings, no timestamps
	bsn_data = np.array([np.array(x[1:]) for x in readings[0:]]) # TODO
			
	# initialize filter
	# (all constructor parameters have defaults, and pykalman supposedly does a
	# good job of estimating them, so we will be lazy until there is a need to
	# define the initial parameters)
	bsn_kfilter = KalmanFilter(
		initial_state_mean = bsn_data[0],
		n_dim_state = len(bsn_data[0]),
		n_dim_obs = len(bsn_data[0]),
		em_vars = 'all'
	)

	# perform parameter estimation and do predictions
	print("Estimating parameters...")
	bsn_kfilter.em(X=bsn_data, n_iter=5, em_vars = 'all')
	print("Creating smoothed estimates...")
	filtered_bsn_data = bsn_kfilter.smooth(bsn_data)[0]

	# re-attach the time steps to observations
	filtered_bsn_data = bind_timesteps_and_data_in_list(
		[x[0:1][0] for x in readings],
		filtered_bsn_data
	)

	# write the data to a new file
	with open(FILTERED_BSN_DATA_FILE_NAME, "wb") as filtered_file:
		filtered_file.write(string.ljust(data_format, 25))
		for filtered_item in filtered_bsn_data:
			print(filtered_item)
			filtered_file.write(struct.pack(data_format, *filtered_item))
		filtered_file.close()

def bind_timesteps_and_data_in_list(timesteps, data):
	bound_list = []
	for i in xrange(len(timesteps)):
		bound_list.append([timesteps[i]] + data[i].tolist())
		
	return bound_list

if __name__ == "__main__":
	main()
