#!/bin/bash
clear
clear
clear


for run in {1..3..1}
do
    training_filename="train""$run"".txt"
    testing_filename="test""$run""_1.txt"
    output_filename="test""$run""_output.txt"

    echo $training_filename
    echo $testing_filename
    echo $output_filename

    ./a.out 30 $training_filename $testing_filename $output_filename
done

