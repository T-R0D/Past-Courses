#include <string>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <exception>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "naive_bayes.cpp"
#include "eigen_space.cpp"

#define PI 3.141592653589793238462643383279502884L
#define REGULARIZATION_FACTOR 0.5
#define IDEAL_NUM_PRINCIPLE_COMPONENTS 16


#define _l_ std::cout<<__LINE__<<std::endl;


const std::string TRAINING_FILE_NAME = "data/zip.train"; // 7291 items
const std::string TEST_FILE_NAME = "data/zip.test"; // 2007 items



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
      std::cout << i << "%% processed." << std::endl;
    }
    i++;

    unsigned classifiedAs = ClassifyObject(object, classSummaries);
    confusionMatrix(object.label, classifiedAs)++;
  }

  return confusionMatrix;
}

Eigen::MatrixXi
PerformPcaClassifications(const std::vector<unsigned>& labelSet,
  const std::vector<ReducedDimClassInfo>& classSummaries,
  const std::vector<ClassificationObject>& testObjects) {

  Eigen::MatrixXi confusionMatrix(labelSet.size(), labelSet.size());
  confusionMatrix.setZero(labelSet.size(), labelSet.size());

  unsigned i = 1;
  unsigned onePercent = testObjects.size() / 100;
  for (const ClassificationObject& object : testObjects) {
    if (i % onePercent == 0) {
      std::cout << (double)i * 100.0 / testObjects.size() << "% processed." << std::endl;
    }
    i++;

    unsigned classifiedAs = PcaClassifyObject(object, classSummaries);
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

  std::cout << "Training Reduced Dimensionality Models..." << std::endl;
  std::vector<ReducedDimClassInfo> reducedClassSummaries; reducedClassSummaries.reserve(10);
  for (unsigned label = 0; label <= 9; ++label) {
    reducedClassSummaries.push_back(
      ComputeClassInfoInPcaDimensionReducedSpace(
        trainingData,
        classSummaries[label],
        16
      )
    );
  }
  
  std::cout << "Classifying Objects..." << std::endl;
  Eigen::MatrixXi confusionMatrix = PerformPcaClassifications(
    {0,1,2,3,4,5,6,7,8,9},
    reducedClassSummaries,
    testData
  );

  std::cout << "Results:" << std::endl;
  std::cout << confusionMatrix << std::endl;

	return 0;
}
