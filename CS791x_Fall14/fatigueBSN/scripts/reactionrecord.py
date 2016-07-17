#!/usr/bin/env python

import time
import struct
import random
import termios
import fcntl
import sys
import os
from math import pow
from include.typodistance import euclideanKeyboardDistance


qwertyKeyboardArrayMissing = [
    ['1','2','3','4','5','6','7','8','9','0']#,
    # ['q','w','e','r','t','y','u','i','o','p','[',']'],
    # ['a','s','d','f','g','h','j','k','l'],
    # ['z','x','c','v','b','n','m']
]


def read_single_keypress():
    """Waits for a single keypress on stdin.

    This is a silly function to call if you need to do it a lot because it has
    to store stdin's current setup, setup stdin for reading single keystrokes
    then read the single keystroke then revert stdin back after reading the
    keystroke.

    Returns the character of the key that was pressed (zero on
    KeyboardInterrupt which can happen when a signal gets handled)

    """
    fd = sys.stdin.fileno()
    # save old state
    flags_save = fcntl.fcntl(fd, fcntl.F_GETFL)
    attrs_save = termios.tcgetattr(fd)
    # make raw - the way to do this comes from the termios(3) man page.
    attrs = list(attrs_save)  # copy the stored version to update
    # iflag
    attrs[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK
                  | termios.ISTRIP | termios.INLCR | termios. IGNCR
                  | termios.ICRNL | termios.IXON)
    # oflag
    attrs[1] &= ~termios.OPOST
    # cflag
    attrs[2] &= ~(termios.CSIZE | termios. PARENB)
    attrs[2] |= termios.CS8
    # lflag
    attrs[3] &= ~(termios.ECHONL | termios.ECHO | termios.ICANON
                  | termios.ISIG | termios.IEXTEN)
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    # turn off non-blocking
    fcntl.fcntl(fd, fcntl.F_SETFL, flags_save & ~os.O_NONBLOCK)
    # read a single keystroke
    try:
        ret = sys.stdin.read(1)  # returns a single character
    except KeyboardInterrupt:
        ret = 0
    finally:
        # restore old state
        termios.tcsetattr(fd, termios.TCSAFLUSH, attrs_save)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags_save)
    return ret


def main():
    random.seed()
    data = []
    try:
        while True:
            print "Ready?"
            reflex_time_list = []
            accuracy_list = []
            for i in xrange(5):
                # choose Random Key
                row = random.randrange(0, len(qwertyKeyboardArrayMissing))
                col = random.randrange(0, len(qwertyKeyboardArrayMissing[row]))
                rand_key = qwertyKeyboardArrayMissing[row][col]
                # choose Random Delay
                time.sleep(random.uniform(0.5, 2))
                print "Press: {0}".format(rand_key)
                x = time.time()
                s = read_single_keypress()
                y = time.time()
                z = euclideanKeyboardDistance(s, rand_key)

                reflex_time_list.append(y-x)
                accuracy_list.append(z)

            # compute mean
            mean_reflex = sum(reflex_time_list)/len(reflex_time_list)
            mean_accuracy = sum(accuracy_list)/len(accuracy_list)

            var_reflex = sum([pow(i - mean_reflex, 2) for i in reflex_time_list])
            var_accuracy = sum([pow(i - mean_accuracy, 2) for i in accuracy_list])
            print (mean_reflex, var_reflex)
            print (mean_accuracy, var_accuracy)
            data.append([time.time(), mean_reflex, var_reflex, mean_accuracy, var_accuracy])
            print ("Press Space To run again or another key to stop and store:")
            if read_single_keypress() != ' ':
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass

    file_ = open(sys.argv[1], 'wb')

    for line in data:
        # data format:
        # timestamp, mean reflex time, reflex time variance, accuracy mean, accuracy variance
        s = struct.pack('ddddd', *line)
        file_.write(s)
    file_.close()


if __name__ == '__main__':
    main()