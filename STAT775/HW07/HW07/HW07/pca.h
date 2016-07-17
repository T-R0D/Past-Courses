#ifndef _PCA_HPP_
#define _PCA_HPP_

#include <vector>
#include <utility>
#include <Eigen\Dense>

struct PcaModel {
  Eigen::MatrixXd rotation_matrix;
  Eigen::RowVectorXd col_means;
};


bool
EigenvectorCompare(
    const std::pair<double, Eigen::VectorXd>& p,
    const std::pair<double, Eigen::VectorXd>& q) {

  return p.first < q.first;
}

Eigen::MatrixXd
GetSortedEigenvectors(const Eigen::MatrixXd& matrix) {
  Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver;
  eigenSolver.compute(matrix, true);

  std::vector<std::pair<double, Eigen::VectorXd>> sortedEigenvectors;
  for (unsigned i = 0; i < eigenSolver.eigenvalues().size(); ++i) {
    if (eigenSolver.eigenvalues().imag()(i) == 0) {
      std::pair<double, Eigen::VectorXd> temp;
      temp.first = eigenSolver.eigenvalues().real()(i);
      temp.second = eigenSolver.eigenvectors().col(i).real();
      sortedEigenvectors.push_back(temp);
    }
  }
  std::sort(sortedEigenvectors.begin(), sortedEigenvectors.end(), EigenvectorCompare);

  Eigen::MatrixXd rowMatrixOfSortedEigenvectors;
  rowMatrixOfSortedEigenvectors.resizeLike(matrix);

  for (unsigned i = 0; i < matrix.cols(); ++i) {
    if (sortedEigenvectors[i].second.rows() == rowMatrixOfSortedEigenvectors.rows()) {	
       rowMatrixOfSortedEigenvectors.col(i) = sortedEigenvectors[i].second;
    } else {
      rowMatrixOfSortedEigenvectors.col(i).setZero();
    }
  }

  int g;

  return rowMatrixOfSortedEigenvectors.transpose();
}

PcaModel
ConstructPcaModel(Eigen::MatrixXd data) {
/**
 *
 */
  Eigen::RowVectorXd mean_vector(data.cols());
  mean_vector.setZero();

  for (int i = 0; i < data.cols(); ++i) {
    for (int j = 0; j < data.rows(); ++j) {
      mean_vector(j) += data(i, j);
    }
  }
  mean_vector /= data.cols();

  Eigen::MatrixXd A = data;
  for (int i = 0; i < A.rows(); ++i) {
    A.row(i) -= mean_vector;
  }
  Eigen::MatrixXd covariance = A * A.transpose();

  Eigen::MatrixXd rotation_matrix = GetSortedEigenvectors(covariance);

  PcaModel model = {
    rotation_matrix,
    mean_vector
  };

  return model;
}

#endif //define _PCA_HPP_