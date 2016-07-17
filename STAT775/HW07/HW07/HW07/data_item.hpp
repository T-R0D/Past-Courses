#ifndef _DATA_ITEM_HPP_
#define _DATA_ITEM_HPP_

#include <fstream>
#include <vector>
#include <utility>
#include <Eigen\Dense>

struct LabeledData {
  std::vector<unsigned> labels;
  Eigen::MatrixXd data;
};

struct DataItem {
  unsigned target;
  Eigen::RowVectorXd point;
};

struct PredicitonItem {
  unsigned target;
  unsigned prediction;
};


Eigen::MatrixXd StlVectorsToEigenMatrix(
  const std::vector<std::vector<double> >& data) {
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

LabeledData 
ReadDataFromFile(const char* file_name) {
/**
 *
 */
  std::fstream fin;
  fin.open(file_name);

  std::vector<unsigned> labels;
  std::vector<std::vector<double>> data;
  double dummy;
  while (fin.good()) {
    fin >> dummy;
    labels.push_back((unsigned) dummy);

    std::vector<double> row;
    unsigned i;
    for (i = 0; i < 256; ++i) {
      fin >> dummy;
      row.push_back(dummy);
    }
    data.push_back(row);
  }

  LabeledData labeledData = {
    labels, StlVectorsToEigenMatrix(data)
  };

  return labeledData;
}

Eigen::MatrixXd GetDataSubset(const LabeledData* labeledData, std::vector<int> labels) {
/**
 *
 */




  return Eigen::MatrixXd(1,1);
}


#endif //define _DATA_ITEM_HPP_