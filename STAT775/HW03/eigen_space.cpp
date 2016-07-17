#ifndef _EIGEN_SPACE_
#define _EIGEN_SPACE_ 1

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <exception>
#include <algorithm>

#include "naive_bayes.cpp"

#define _l_ std::cout<<__LINE__<<std::endl;

typedef struct {
  Eigen::VectorXd longMean;
  Eigen::MatrixXd transformationMatrix;
  ClassInfo ci;
} ReducedDimClassInfo;


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
      sortedEigenvectors.push_back(
        {
          eigenSolver.eigenvalues().real()(i),
          eigenSolver.eigenvectors().col(i).real()
        }
      );
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

  return rowMatrixOfSortedEigenvectors.transpose();
}


ReducedDimClassInfo
ComputeClassInfoInPcaDimensionReducedSpace(
    const std::vector<ClassificationObject>& data,
    const ClassInfo& cartesianClassInfo,
    const unsigned reducedDimensionality) {

  // create the space transformation matrix
  unsigned originalDimensionality = cartesianClassInfo.mean.size();
  Eigen::MatrixXd eigenspaceTransformMatrix(
    reducedDimensionality,
    originalDimensionality
  );

  eigenspaceTransformMatrix =
    (GetSortedEigenvectors(cartesianClassInfo.covariance)).topLeftCorner(
      reducedDimensionality,
      originalDimensionality
    );

  // transform all of the data to the new space
  std::vector<ClassificationObject> reducedData;
  reducedData.reserve(data.size());
  for (unsigned i = 0; i < data.size(); ++i) {
    ClassificationObject temp;
    temp.label = data[i].label;

    temp.features.resize(reducedDimensionality);
    temp.features = eigenspaceTransformMatrix * data[i].features;

    reducedData.push_back(temp);
  }

  // compute the new class info
  ReducedDimClassInfo info;
  info.longMean = cartesianClassInfo.mean;
  info.transformationMatrix = eigenspaceTransformMatrix;
  info.ci = ComputeClassInfo(
    reducedData,
    cartesianClassInfo.label,
    reducedDimensionality
  );

  return info;
}

unsigned
PcaClassifyObject(const ClassificationObject& object, const std::vector<ReducedDimClassInfo>& classSummaries) {
  unsigned mostLikelyClass = classSummaries.front().ci.label;
  double highestProbability = 0.0;

  for (const ReducedDimClassInfo& classSummary : classSummaries) {
    Eigen::VectorXd scaledFeatureVector = object.features - classSummary.longMean;
    Eigen::VectorXd reducedFeatureVector = classSummary.transformationMatrix * scaledFeatureVector;
    double probability = GaussianPdf(reducedFeatureVector, classSummary.ci) * classSummary.ci.prior;
    if (probability > highestProbability) {
      mostLikelyClass = classSummary.ci.label;
      highestProbability = probability;
    }
  }

std::cout<<highestProbability<<std::endl;

  return mostLikelyClass;
}

#endif //_EIGEN_SPACE_