/**
 *
 */
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>

#include "random_discriminant.hpp"
#include "data_item.hpp"
#include "pca.h"

#if 0
const char* TRAIN_FILE_NAME = "../DataSets/zip.data/zip.train";
const char* TEST_FILE_NAME = "../DataSets/zip.data/zip.test";
#else
const char* TRAIN_FILE_NAME =
  "C:/Users/Terence/Documents/GitHub/STAT775/DataSets/zip.data/zip.train";
const char* TEST_FILE_NAME =
  "C:/Users/Terence/Documents/GitHub/STAT775/DataSets/zip.data/zip.test";
#endif // if for environment





//LabeledData ReadDataFromFile(const char* file_name, const int dimensionality) {
///**
// *
// */
//  FILE* file = fopen(file_name, "r");
//  if (file == NULL) {
//    printf("Failed to read from file: %s\n", file_name);
//    exit(0);
//  }
//
//  std::vector<int> labels;
//  std::vector<std::vector<double> > data;
//
//  double dummy;
//  int read_result = fscanf(file, "%lf", &dummy);
//  while (read_result >= 1) {
//    labels.push_back((int) dummy);
//
//    std::vector<double> temp;
//    unsigned i;
//    for (i = 0; i < dimensionality; ++i) {
//      fscanf(file, " %lf", &dummy);
//      temp.push_back(dummy);
//    }
//    data.push_back(temp);
//
//    read_result = fscanf(file, " %lf", &dummy);
//  }
//
//  LabeledData labeledData = {labels, StlVectorsToEigenMatrix(data)};
//
//  return labeledData;
//}

int main(int argc, char** argv) {
   printf("%s\n%s\n", TRAIN_FILE_NAME, TEST_FILE_NAME);

  int i = 0;

  LabeledData data = 
    ReadDataFromFile(TRAIN_FILE_NAME);

  ConstructPcaModel(data.data);

  //std::cout << data.data.row(0).head(10);

  //Eigen::MatrixXd test_0(2, 2); test_0 << -2.0, -1.0,
  //                                        -2.0, -1.0;
  //Eigen::MatrixXd test_1(2, 2); test_1 <<  1.0,  1.0,
  //                                         1.0,  1.0;

  //DiscriminantModel discriminant_model;
  //InitDiscriminantModel(&discriminant_model, 2, &test_0, &test_1);
  //PrintDiscriminantModel(&discriminant_model, stdout, true);
  //std::cout << "Class 0 predictions:\n" << GetDiscriminantPredictions(&discriminant_model, &test_0) << std::endl;
  //std::cout << "Class 1 predictions:\n" << GetDiscriminantPredictions(&discriminant_model, &test_1) << std::endl;

  //discriminant_model.projection_line << 0, 1;
  //std::cout << ((discriminant_model.projection_line).dot(test_0.row(0))) << std::endl;


  //LabeledData train = ReadDataFromFile(TRAIN_FILE_NAME, 256);
  //std::cout << train.data.row(0) << std::endl;

  //std::cout << train.labels[0] << ' ' << train.labels[1] << ' ' << train.labels[2];

  return 0;
}