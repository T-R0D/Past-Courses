#!/bin/bash

# make and move the sequential code
cd sequential_matrix_multiply
make
mv /home/thenriod/CS615/PA03/sequential_matrix_multiply/Sequential_Matrix_Multiply /home/thenriod/CS615/PA03/
cd ..

# make and move the parallel code
cd modified_summa
make
mv /home/thenriod/CS615/PA03/modified_summa/Modified_SUMMA /home/thenriod/CS615/PA03/
cd ..

# make and move the node reporting program
cd node_reporter
make
mv /home/thenriod/CS615/PA03/node_reporter/Node_Report /home/thenriod/CS615/PA03/
cd ..
