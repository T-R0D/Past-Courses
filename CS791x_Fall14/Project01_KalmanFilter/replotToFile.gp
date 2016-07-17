# This 'macro' comes from
# "Gnuplot in Action: Understanding Data Wth Graphs"
# by Philipp K. Janert

set terminal push	# save the current terminal settings
set terminal png 	# change terminal to PNG
set output "$0" 	# set the output filename to the first option
replot 				# repeat the most recent plot command
set output 			# restore output to interactive mode
set terminal pop 	# restore the terminal
