#ifndef _NAIVE_BAYES_
#define _NAIVE_BAYES_ 1

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <exception>
#include <algorithm>

#define PI 3.141592653589793238462643383279502884L

#define _l_ std::cout<<__LINE__<<std::endl;


typedef struct {
  unsigned label;
  Eigen::VectorXd features;
} ClassificationObject;

typedef struct {
  unsigned label;
  double prior;
  Eigen::VectorXd mean;
  Eigen::MatrixXd covariance;
  double covarianceDeterminant;
  Eigen::MatrixXd covarianceInverse;
} ClassInfo;

ClassInfo
ComputeClassInfo(
    const std::vector<ClassificationObject>& data,
    const unsigned classLabel,
    const unsigned featureDim) {

  ClassInfo classInfo;
  classInfo.label = classLabel;
  classInfo.mean.setZero(featureDim);
  classInfo.covariance.setZero(featureDim, featureDim);

  // compute mean and prior probability at once
  std::vector<ClassificationObject> subset;
  for (const ClassificationObject& classificationObject : data) {
    if (classificationObject.label == classLabel) {
      classInfo.mean += classificationObject.features;
      subset.push_back(classificationObject);
    }
  }
  classInfo.mean /= subset.size();
  classInfo.prior = (double) subset.size() / (double) data.size();

  // covariance matrix
  // Note: I use a nifty trick that is simpler in code (not sure if
  //       simpler computationally). Let $P_i = x_i - mu$, where $i$
  //       corresponds to a particular observation vector. Then
  //       $A$ is the concatenation of all the $P_i$ as column
  //       vectors ($A = [P_1 ... P_m]$). Then
  //       $\frac{1}{m} * A * A^t$ results in the same computations
  //       that create the covariance matrix as more traditional
  //       formulae.
  Eigen::MatrixXd A(featureDim, subset.size());
  for (unsigned j = 0; j < subset.size(); ++j) {
    A.col(j) = subset[j].features - classInfo.mean;
  }

  classInfo.covariance = A * A.transpose();
  classInfo.covariance /= (double) subset.size();

  classInfo.covarianceDeterminant = classInfo.covariance.determinant();
  classInfo.covarianceInverse = classInfo.covariance.inverse();

  return classInfo;
}

double
GaussianPdf(
  const Eigen::VectorXd& testFeatureVector,
  const ClassInfo& classSummary) {

  const Eigen::VectorXd& x = testFeatureVector;
  const Eigen::VectorXd& mu = classSummary.mean;
  const double& sigmaDet = classSummary.covarianceDeterminant;
  const Eigen::MatrixXd& sigmaInv = classSummary.covarianceInverse;

  double scalingFactor = 1.0 / sqrt(pow(2.0 * PI, mu.size()) * sigmaDet);
  double exponent = -0.5 * (((x - mu).transpose() * sigmaInv).dot((x - mu)));

  return scalingFactor * exp(exponent);
}

unsigned
ClassifyObject(const ClassificationObject& object, const std::vector<ClassInfo>& classSummaries) {
  unsigned mostLikelyClass = 0;
  double highestProbability = classSummaries.front().label;

  for (const ClassInfo& classSummary : classSummaries) {
    double probability = GaussianPdf(object.features, classSummary) * classSummary.prior;
    if (probability > highestProbability) {
      mostLikelyClass = classSummary.label;
      highestProbability = probability;
    }
  }

  return mostLikelyClass;
}

#endif //_NAIVE_BAYES_