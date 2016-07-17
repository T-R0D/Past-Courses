/**
 *
 */
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>

const char* TRAIN_FILE_NAME = "../DataSets/zip.data/zip.train";
const char* TEST_FILE_NAME = "../DataSets/zip.data/zip.test";

typedef struct {
  std::vector<int> labels;
  Eigen::MatrixXd data;
} LabeledData;

Eigen::MatrixXd StlVectorsToEigenMatrix(const std::vector<std::vector<double> >& data) {
/**
 *
 */
  Eigen::MatrixXd matrix(data.size(), data[0].size());

  unsigned i;
  unsigned j;
  for (i = 0; i < data.size(); ++i) {
    for (j = 0; j < data[0].size(); ++j) {
      matrix(i, j) = data[i][j];
    }
  }

  return matrix;
}

Eigen::MatrixXd GetDataSubset(const LabeledData* labeledData, std::vector<int> labels) {
/**
 *
 */





}

LabeledData ReadDataFromFile(const char* file_name, const int dimensionality) {
/**
 *
 */
  FILE* file = fopen(file_name, "r");
  if (file == NULL) {
    printf("Failed to read from file: %s\n", file_name);
    exit(0);
  }

  std::vector<int> labels;
  std::vector<std::vector<double> > data;

  double dummy;
  int read_result = fscanf(file, "%lf", &dummy);
  while (read_result >= 1) {
    labels.push_back((int) dummy);

    std::vector<double> temp;
    unsigned i;
    for (i = 0; i < dimensionality; ++i) {
      fscanf(file, " %lf", &dummy);
      temp.push_back(dummy);
    }
    data.push_back(temp);

    read_result = fscanf(file, " %lf", &dummy);
  }

  LabeledData labeledData = {labels, StlVectorsToEigenMatrix(data)};

  return labeledData;
}

int main(int argc, char** argv) {
  // printf("%s\n%s\n", TRAIN_FILE_NAME, TEST_FILE_NAME);

  LabeledData train = ReadDataFromFile(TRAIN_FILE_NAME, 256);
  std::cout << train.data.row(0) << std::endl;

  std::cout << train.labels[0] << ' ' << train.labels[1] << ' ' << train.labels[2];

  return 0;
}