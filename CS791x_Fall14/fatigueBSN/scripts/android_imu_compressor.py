import sys
import struct

""" Takes data from the Android IMU app and turns it into binary data.

Data comes in as csv, data points will be turned into the format:

	Time Stamp    Accelerometer     Gyroscope
					x  y  z          x  y  z
	=========================================
	    0           1  2  3          4  5  6
"""

ANDROID_IMU_DATA_FORMAT_STRING = 'ddddddd'
HEADER_SIZE = 25


def main():
	input_file_name = sys.argv[1]
	output_file_name = sys.argv[2]

	with open(output_file_name, "wb") as out_file:
		# write the format header
		out_file.write(
			ANDROID_IMU_DATA_FORMAT_STRING.ljust(HEADER_SIZE, ' ')
		)

		with open(input_file_name, "r") as in_file:
			for line in in_file: # ??????????????? Is Ok? ??????????????????
				clean_data = line_to_clean_data(line)
				if clean_data:
					out_file.write(
						struct.pack(ANDROID_IMU_DATA_FORMAT_STRING, *clean_data)
					)
			in_file.close()
		out_file.close()


def line_to_clean_data(line):
	if not '4,' in line:
		return None
	else:
		items_as_text = line.split(",")

		if len(items_as_text) < 13: # expected number of items in line
			return None

		item_values = [float(x) for x in items_as_text]

		data_items = [
			item_values[0],  # time stamp
			item_values[2],  # accelerometer x
			item_values[3],  # accelerometer y
			item_values[4],  # accelerometer z
			item_values[6],  # gyroscope x
			item_values[7],  # gyroscope y
			item_values[8]   # gyroscope z
		]

		return data_items


if __name__ == '__main__':
	main()
