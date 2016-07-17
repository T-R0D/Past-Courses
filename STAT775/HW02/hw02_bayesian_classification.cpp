#include <string>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <exception>

#include <iostream>
#include <fstream>

#define PI 3.141592653589793238462643383279502884L
#define REGULARIZATION_FACTOR 0.5

const std::string TRAINING_FILE_NAME = "data/zip.train"; // 7291 items
const std::string TEST_FILE_NAME = "data/zip.test"; // 2007 items

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


std::vector<ClassificationObject>
ReadData(
  const std::string& fileName,
  const unsigned featureDim,
  const unsigned numItems) {

  std::ifstream fin;
  fin.clear(); fin.open(fileName.c_str());

  if (!fin.good()) {
    throw std::exception();
  }

  std::vector<ClassificationObject> data; data.reserve(numItems);

  for (unsigned i = 0; i < numItems; ++i) {
    ClassificationObject classificationObject;
    double dummy;
    fin >> dummy;
    classificationObject.label = (unsigned) dummy;

    classificationObject.features.resize(featureDim);
    for (unsigned j = 0; j < featureDim; ++j) {
      fin >> classificationObject.features(j);
    }
    data.push_back(classificationObject);
  }

  return data;
}

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
  if (classInfo.covariance.determinant() - 0.1 <= 0.0) {
    Eigen::MatrixXd regularizationMatrix;
    regularizationMatrix.setIdentity(featureDim, featureDim);
    regularizationMatrix *= REGULARIZATION_FACTOR;
    classInfo.covariance += regularizationMatrix;
  }
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

Eigen::MatrixXi
PerformClassifications(const std::vector<unsigned>& labelSet,
  const std::vector<ClassInfo>& classSummaries,
  const std::vector<ClassificationObject>& testObjects) {

  Eigen::MatrixXi confusionMatrix(labelSet.size(), labelSet.size());
  confusionMatrix.setZero(labelSet.size(), labelSet.size());

  unsigned i = 1;
  unsigned onePercent = testObjects.size() / 100;
  for (const ClassificationObject& object : testObjects) {
    if (i % onePercent == 0) {
      std::cout << i << "% processed." << std::endl;
    }
    i++;

    unsigned classifiedAs = ClassifyObject(object, classSummaries);
    confusionMatrix(object.label, classifiedAs)++;
  }

  return confusionMatrix;
}


int
main(const int argc, const char** argv) {
  std::cout << "Reading Data..." << std::endl;
  std::vector<ClassificationObject> trainingData = ReadData(
    TRAINING_FILE_NAME,
    256,
    7291
  );
  std::vector<ClassificationObject> testData = ReadData(
    TEST_FILE_NAME,
    256,
    2007
  );

  std::cout << "Training Models..." << std::endl;
  std::vector<ClassInfo> classSummaries; classSummaries.reserve(10);
  for (unsigned label = 0; label <= 9; ++label) {
    classSummaries.push_back(ComputeClassInfo(trainingData, label, 256));
  }

  std::cout << "Classifying Objects..." << std::endl;
  Eigen::MatrixXi confusionMatrix = PerformClassifications(
    {0,1,2,3,4,5,6,7,8,9},
    classSummaries,
    testData
  );

  std::cout << "Results:" << std::endl;
  std::cout << confusionMatrix << std::endl;

	return 0;
}
