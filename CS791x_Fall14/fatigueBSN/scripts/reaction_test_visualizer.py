#!/usr/bin/env python

from __future__ import print_function
import struct

import pylab as pl
from bsn_data_point import NON_FATIGUE_LABEL
from bsn_data_point import FATIGUE_LABEL


LUKE_FILE = '/home/t/Desktop/fatigueBSN/fatigue_test_data/Luke/12_03_2240_reaction.dat'
TERENCE_FILE = '/home/t/Desktop/fatigueBSN/fatigue_test_data/Terence/12_03_2002_reaction.dat'

def main():
	data = read_reaction_data_into_list('/home/t/Desktop/fatigueBSN/fatigue_test_data/Luke/12_03_2240_reaction.dat')

	for item in generate_labels_with_times(data, 1.1):  # TODO: find a better threshold
		print("{}: {}".format(item[0], item[1]))

	# data format:
	# timestamp, mean reflex time, reflex time variance, accuracy mean, accuracy variance
	reaction_time = [x[1] for x in data]
	accuracy = [1.0 - x[3] for x in data]

	pl.plot(range(len(data)), reaction_time, label='reaction time')
	pl.plot(range(len(data)), accuracy, label='reaction accuracy')
	pl.xlabel("Time Step")
	pl.ylabel("Reaction Value")
	pl.title('Reaction Test Data')
	legend = pl.legend(loc='best', ncol=2, shadow=None)
	legend.get_frame().set_facecolor('#00FFCC')
	pl.show()


def generate_labels_with_times(reaction_data, reaction_time_threshold):
	labels = []
	label = NON_FATIGUE_LABEL
	for item in reaction_data:
		if item[1] > reaction_time_threshold:
			label = FATIGUE_LABEL
		labels.append((item[0], label))
	return labels


def read_reaction_data_into_list(file_name):
	with open(file_name) as file_:
		data = []
		for packed_struct in unpack_structs_from_file(file_, struct.calcsize('ddddd')):
			if packed_struct is not None:
				data_tuple = struct.unpack('ddddd', packed_struct)
				data.append(data_tuple)
		file_.close()
	return data


def unpack_structs_from_file(file_, struct_size):
	while True:
		struct_data = file_.read(struct_size)
		if struct_data:
			yield struct_data
		else:
			break
	yield None


if __name__ == '__main__':
	main()