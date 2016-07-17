plot [0:3900] [-4:4] "$0" using 2 title "Odometry" with lines, \
	"$0" using 3 title "IMU" with lines, \
	"$0" using 4 title "KalmanFilter" with lines;
set title "Angular Heading over Time";
set xlabel "Time Step (0.001s)"; show xlabel;
set ylabel "Heading (radians)"; show ylabel;