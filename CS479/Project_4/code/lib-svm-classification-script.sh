#!/bin/bash
clear; clear;

# configurations
data_src_dir="prepared_data/"
svm_results_dir="svm_results/"

#variables
name_base=""
param_c=1
kernel_type="polynomial"
data_file="nonameyet"
trained_model_file="nonameyet"
results_file="nanameyet"

g++ -o merge-4-libsvm merge-4-libsvm.cpp
g++ -o svm-classification project_4_svm_classification_driver.cpp svm.cpp
g++ -o bayesian-classification -I /home/thenriod/Desktop/cpp_libs/Eigen_lib project_4_bayes_classification_driver.cpp bayes_classifier.cpp

bash data_preparation_script.sh





# train several SVMs with polynomial kernels and RBF kernels
    echo ""
    echo "==========================================="
    echo "|         Training Polynomial SVMs        |"
    echo "==========================================="
    echo ""
kernel_type="polynomial"
for param_c in {1..5..1}
do
    for i in {1..3..1}
    do
        name_base="$svm_results_dir""$kernel_type""_""$param_c""_"

        trained_model_file="$name_base""trained_svm.txt"
        results_file="$name_base""results.txt"

        echo ""
        echo "==============="
        echo "| Training... |"
        echo "==============="
        echo ""
        echo "type = $kernel_type  c = $param_c"
        data_file="$data_src_dir""training_$i""_scaled.txt"
        ./svm-train -t 2 -g 0.5 -c $param_c $data_file $trained_model_file
    
        echo ""
        echo "================="
        echo "| Predicting... |"
        echo "================="
        echo ""
        data_file="$data_src_dir""testing_$i""_scaled.txt"
        ./svm-predict $data_file $trained_model_file $results_file
    done
done

