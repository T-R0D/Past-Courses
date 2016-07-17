#!/usr/bin/env python

""" Merges data into a format ready for data processing.

    Final Data Format (?):
    Time, label, Low-Alpha, High-Alpha, Heart Rate, IMU angle
"""

import struct

# import reaction_test_visualizer as rtv


LUKE_FILES = [
    '../fatigue_test_data/Luke/12_03_2240_reaction.dat',
    '../fatigue_test_data/Luke/12_03_2202_heart.dat',
    '../fatigue_test_data/Luke/12_03_2227_mind.dat',
    '../fatigue_test_data/Luke/12_03_2002_cleaned_imu.dat',
    '../fatigue_test_data/Luke/12_03_2002_merged.dat'
]
TERENCE_FILES = [
    '../fatigue_test_data/Terence/12_03_2002_reaction.dat',
    '../fatigue_test_data/Terence/12_03_2002_heart.dat',
    '../fatigue_test_data/Terence/12_03_2002_mind.dat',
    '../fatigue_test_data/Terence/12_03_2002_cleaned_imu.dat',
    '../fatigue_test_data/Terence/12_03_2002_merged.dat'
]

def main():
    # files = TERENCE_FILES
    files = LUKE_FILES

    # load files
    # load IMU
    imu_file = files[3]
    imu_data = []

    # Mindwave
    mindwave_file = files[2]

    # Heartrate depending on the length of sys.argc
    heart_file = files[1]

    # Read data from IMU file
    imu_data, fmt_imu = unpack_binary_data_into_list(imu_file)
    mindwave_data, fmt_mind = unpack_binary_data_into_list(mindwave_file)
    heart_data, fmt_heart = unpack_binary_data_into_list(heart_file)

    imu_data = [[x[0], sum(x[4:7])] for x in imu_data] # integrate
    mindwave_data = [[x[0], x[3], x[4]] for x in mindwave_data]
    heart_data = [[x[0], float(x[1])] for x in heart_data]
    merged_data = interpolate_data(imu_data, heart_data, mindwave_data)

    # times_and_lables = rtv.generate_labels_with_times(reaction_data=reaction_data, reaction_time_threshold=.1)
    # tagged_data = tag_data(files[0], merged_data)

    # Save data
    print len(merged_data)
    f = open(files[4], 'wb')
    fmt_merge = 'd' * len(merged_data[0])
    f.write(fmt_merge.ljust(25, ' '))
    for row in merged_data:
        f.write(struct.pack(fmt_merge, *row))
        # print row
    f.close()


def unpack_binary_data_into_list(file_name):
    data_points = []
    file_ = open(file_name, 'rb')
    fmt = file_.read(25).rstrip()
    struct_size = struct.calcsize(fmt)

    for packed_struct in packed_structs_from_file(file_, struct_size):
        if packed_struct is not None:
            data_tuple = struct.unpack(fmt, packed_struct)
            data_points.append(list(data_tuple))
    file_.close()
    return data_points, fmt


def packed_structs_from_file(file_, struct_size):
    while True:
        struct_data = file_.read(struct_size)
        if struct_data:
            yield struct_data
        else:
            break
    yield None


def interpolate_data(leader_data, *data_lists):
    # maps from list (0 for leader_data or varargs_index + 1) to indices of
    # first and last data points to be used
    merge_list = []
    for i, row in enumerate(leader_data):
        current_time = row[0]
        merge_list.append(row)
        for arg in data_lists:
            # Find local indices around data
            low, high = local_indicies(current_time, arg)
            merge_list[i] += interpolate(current_time, arg[low], arg[high])
    return merge_list


def local_indicies(lead_step, child_steps):
    """
    Returns a tuple of indices surrounding the lead_step
    """
    imin = 0
    imax = len(child_steps) - 1
    mid = 0
    while imin < imax:
        mid = (imax + imin)/2
        if(child_steps[mid][0] < lead_step):
            imin = mid + 1
        else:
            imax = mid
    if child_steps[mid][0] < lead_step:
        return (mid, mid+1)
    return mid-1,mid


def interpolate(time, low_val_list, high_val_list):
    """
    Returns a list of interpolated values from the low and high list
    """
    d_t = high_val_list[0] - low_val_list[0]
    x1 = low_val_list[0]
    result = []
    for y1,y2 in map(lambda a,b: (a,b),low_val_list[1:], high_val_list[1:]):
        d_y = y2 - y1
        m = d_y/d_t
        b = y1 - m*x1
        result.append(m * time + b)
    return result


# def tag_data(reaction_data_file_name, data):
#     reaction_data = rtv.generate_labels_with_times(
#         rtv.read_reaction_data_into_list(reaction_data_file_name),
#         1.1
#     )
#     tagged_data = []
#     j = 0
#     for item in data:
#         if reaction_data[j + 1] < item[0]:
#             j += 1
#         tagged_data.append((reaction_data[j], item[2:]))
#
#     return tagged_data


if __name__ == '__main__':
    main()
    # x = [[1.2,1.1,2.1], [1.5,3.2,4.1], [1.8, 4.2,6.5], [2.1,7.0,1.2], [2.5,9.1,1.6], [2.7,10,11.1]]
    # y = [[.1,1,2,3],[.2,1,2,3],[.3,1,2,3],[.4,1,2,3],[.5,1,2,3],[.6,1,2,3],[.7,1,2,3],[.8,1,2,3],[.9,1,2,3],[1.0,1,2,3],[1.1,1,2,3],[1.2,1,2,3],[1.3,1,2,3]]
    # print interpolate_data(y,x,x)
    # low, high = local_indicies(y,x)
    # print low, high
    # print interpolate(y, x[low], x[high])