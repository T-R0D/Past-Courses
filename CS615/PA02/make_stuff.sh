#!/bin/bash

# make and move the sequential code
cd sequential
make
mv /home/thenriod/CS615/PA02/sequential/Mandelbrot_sequential /home/thenriod/CS615/PA02/
cd ..

# make and move the sequential code
cd static
make
mv /home/thenriod/CS615/PA02/static/Mandelbrot_static /home/thenriod/CS615/PA02/
cd ..

# make and move the sequential code
cd dynamic
make
mv /home/thenriod/CS615/PA02/dynamic/Mandelbrot_dynamic /home/thenriod/CS615/PA02/
cd ..

# make and move the node reporting program
cd node_reporter
make
mv /home/thenriod/CS615/PA02/node_reporter/Node_Report /home/thenriod/CS615/PA02/
cd ..
