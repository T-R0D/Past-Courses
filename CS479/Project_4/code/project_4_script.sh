#!/bin/bash
clear; clear;

# configurations
data_src_dir="prepared_data/"
svm_results_dir="svm_results/"
bayes_results_dir="bayesian_results/"
num_features=30
num_data_sets=3

#variables
name_base=""
param_c=1
kernel_type=1
training_file="nonameyet"
testing_file="nonameyet"

g++ -o merge-4-libsvm merge-4-libsvm.cpp
g++ -o svm-classification project_4_svm_classification_driver.cpp svm.cpp
g++ -o bayesian-classification -I /home/thenriod/Desktop/cpp_libs/Eigen_lib project_4_bayes_classification_driver.cpp bayes_classifier.cpp

bash data_preparation_script.sh

for i in {1..3..1}
do
    training_file="$data_src_dir""training_$i""_scaled.txt"
    testing_file="$data_src_dir""testing_$i""_scaled.txt"

    # do svm runs
    for c in {100..1000..100}
    do
        for gamma in {3..8..1}
        do
            kernel_type=1 # polynomial
            ./svm-classification $num_features $kernel_type $c "0.$gamma" $training_file $testing_file $svm_results_dir

            kernel_type=2 # RBF
            ./svm-classification $num_features $kernel_type $c "0.$gamma" $training_file $testing_file $svm_results_dir
        done
    done

    # do bayesian runs
    ./bayesian-classification $num_features $training_file $testing_file "$bayes_results_dir""set_$i""_results.txt"
done

