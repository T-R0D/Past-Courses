#!/bin/bash
clear; clear;

# configurations
raw_dir="raw_data/"
prepared_dir="prepared_data/"
num_features=30

#variables
label_file=""
value_file=""
prepared_file=""

mkdir $prepared_dir

# setup training data
echo "Preparing training data..."
for i in {1..3..1}
do
    label_file="$raw_dir""TtrPCA_0""$i"".txt"
    value_file="$raw_dir""trPCA_0""$i"".txt"
    prepared_file="$prepared_dir""training_""$i"".txt"
    ./merge-4-libsvm $num_features $label_file $value_file $prepared_file
done

cat "$prepared_dir""training_1.txt" "$prepared_dir""training_2.txt" "$prepared_dir""training_3.txt" > "$prepared_dir""all_training_data.txt"

#setup testing data
echo "Preparing testing data..."
for i in {1..3..1}
do
    #validation data
    label_file="$raw_dir""TvalPCA_0""$i"".txt"
    value_file="$raw_dir""valPCA_0""$i"".txt"
    prepared_file="$prepared_dir""validation_""$i"".txt"
    ./merge-4-libsvm $num_features $label_file $value_file $prepared_file

    #testing data
    label_file="$raw_dir""TtsPCA_0""$i"".txt"
    value_file="$raw_dir""tsPCA_0""$i"".txt"
    prepared_file="$prepared_dir""test_""$i"".txt"
    ./merge-4-libsvm $num_features $label_file $value_file $prepared_file
done

for i in {1..3..1}
do
    cat "$prepared_dir""test_""$i"".txt" "$prepared_dir""validation_""$i"".txt" > "$prepared_dir""testing_""$i"".txt"
done

cat "$prepared_dir""testing_1.txt" "$prepared_dir""testing_2.txt" "$prepared_dir""testing_3.txt" > "$prepared_dir""all_testing_data.txt"

# scale the data
echo "Scaling data..."
for i in {1..3..1}
do
    ./svm-scale -s "$prepared_dir""scaling_range_$i.txt" "$prepared_dir""training_$i.txt" > "$prepared_dir""training_$i""_scaled.txt"
    ./svm-scale -r "$prepared_dir""scaling_range_$i.txt" "$prepared_dir""testing_$i.txt" > "$prepared_dir""testing_$i""_scaled.txt"
done

./svm-scale -s "$prepared_dir""scaling_range_all.txt" "$prepared_dir""all_training_data.txt" > "$prepared_dir""all_training_data_scaled.txt"
./svm-scale -r "$prepared_dir""scaling_range_all.txt" "$prepared_dir""all_testing_data.txt" > "$prepared_dir""all_testing_data_scaled.txt"

echo "Data preparation complete."
echo ""
