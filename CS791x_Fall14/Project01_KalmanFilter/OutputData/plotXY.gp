plot [-15:15] [-5:20] "$0" using 2:3 title "Odometry" with lines, \
	"$0" using 4:5 title "GPS" with lines, \
	"$0" using 6:7 title "KalmanFilter" with lines;
set title "Estimation of the Robot's Circular Path";
set xlabel "X position (m)"; show xlabel;
set ylabel "Y position(m)"; show ylabel;