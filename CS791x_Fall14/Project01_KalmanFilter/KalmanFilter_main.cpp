#ifndef __KALMANFILTER_MAIN_CPP__
#define __KALMANFILTER_MAIN_CPP__

#include <cstring>
#include <string>
#include <cstdio>
#include <vector>

#include <cmath>

#include <iostream>
#include <fstream>
#include <iomanip>

#include <Eigen/Dense>

#include "KalmanFilter.hpp"


#define b printf("Line: %d\n", __LINE__);


const int STATE_DIMENSIONS = 5;
const int MEASUREMENT_DIMENSIONS = 5;
const int NUM_MEASUREMENTS = 6;

const double DELTA_T = 0.001;
const double LINEAR_VELOCITY = 0.14;
const double ANGULAR_VELOCITY = LINEAR_VELOCITY * tan(0);
const double ARBITRARY_VARIANCE = 0.1;

/* Given data, resulting matrix arrangement:
        Measurements:           Covariances:
    0    odometer.x              Something arbitrary
    1    GPS.x                   GPS.x covariance
    2    odometer.y              Something arbitrary
    3    GPS.y                   GPS.y covariance
    4    odometer.\theta         Something arbitrary
    5    IMU.\theta              IMU.\theta covariance
 */
const int ODOM_X = 0;
const int ODOM_Y = 2;
const int ODOM_T = 4;
const int GPS_X = 1;
const int GPS_Y = 3;
const int IMU_T = 5;
const int X_ESTIMATE = 0;
const int Y_ESTIMATE = 1;
const int V_ESTIMATE = 2;
const int T_ESTIMATE = 3;
const int W_ESTIMATE = 4;

typedef struct {
    std::string dataFilename;
    std::string xyFilename;
    std::string tFilename;
    std::string vFilename;
    std::string wFilename;
    double stateCovarianceModifier;
} Configurations;


typedef struct {
    Eigen::VectorXd measurements;
    Eigen::VectorXd variances;
} DataItem;

Configurations processCmdLineArgs(int pArgc, char** pArgs);

std::vector<DataItem> readDataFile(const std::string& dataFileName);

DataItem readDataLine(std::ifstream& fin);

std::vector<KalmanFilter::KalmanState> applyKalmanFilterToData(
    Configurations& pConfigurations,
    std::vector<DataItem>& pData);

Eigen::MatrixXd computeUpdatedNaturalModelMatrix(
    int pStateDimensionality,
    double pTimeStepMagnitude,
    Eigen::VectorXd& pPreviousState);

Eigen::MatrixXd computeUpdatedStateNoiseCovariance(
    int pStateDimensionality,
    double pXPosCovariance,
    double pYPosCovariance);

Eigen::MatrixXd computeUpdatedObservationNoiseCovariance(
    int pObservationDimensionality,
    double pXPosCovariance, 
    double pYPosCovariance,
    double pTPosCovariance);

Eigen::VectorXd reduceDataToInputMeasurements(
    Eigen::VectorXd& pMeasurementData,
    int pNewVectorSize,
    const std::vector<std::pair<int, int>>& pDataIndices);

void writeDataToFile(
    std::string pFileName,
    std::vector<DataItem>& pInputData,
    const std::vector<std::string>& pDataColumnLabels,
    const std::vector<int>& pDataIndices,  
    std::vector<KalmanFilter::KalmanState>& pKalmanOutputData,
    const std::vector<std::string>& pOutputColumnLabels,
    const std::vector<int>& pOutputIndices);

void testSimpleKalmanFilter();



int main(int argc, char** argv) {
    Configurations configurations;

    try {
        puts("Initializing program...");
        configurations = processCmdLineArgs(argc, argv);

        puts("Reading data...");
        std::vector<DataItem> data = readDataFile(configurations.dataFilename);

        puts("Applying KalmanFilter...");
        std::vector<KalmanFilter::KalmanState> results =
            applyKalmanFilterToData(configurations, data);

        puts("Writing results to file...");
        writeDataToFile(
            configurations.xyFilename,
            data,
            {"Odom_X", "Odom_Y", "GPS_X", "GPS_Y"},
            {ODOM_X, ODOM_Y, GPS_X, GPS_Y},
            results,
            {"X-estimate", "Y-estimate"},
            {X_ESTIMATE, Y_ESTIMATE}
        );
        writeDataToFile(
            configurations.tFilename,
            data,
            {"Odom_T", "IMU_T"},
            {ODOM_T, IMU_T},
            results,
            {"T-estimate",},
            {T_ESTIMATE}
        );
        writeDataToFile(
            configurations.vFilename,
            data,
            {},
            {},
            results,
            {"v-estimate"},
            {V_ESTIMATE}
        );
        writeDataToFile(
            configurations.wFilename,
            data,
            {},
            {},
            results,
            {"w-estimate",},
            {W_ESTIMATE}
        );

    } catch (std::exception& e) {
        puts("An error occured - program failure!");
        puts(e.what());
    }

    return 0;
}


Configurations processCmdLineArgs(int pArgc, char** pArgs) {
    Configurations configurations;
    
    if (pArgc < (3 + 1)) {
        throw std::exception();
    }

// TODO: use strtok for cleaner arg parsing

    for (int i = 1; i < pArgc; i += 2) {
        if (strcmp(pArgs[i], "-data") == 0) {
            configurations.dataFilename = pArgs[i + 1];
        } else if (strcmp(pArgs[i], "-xy") == 0) {
            configurations.xyFilename = pArgs[i + 1];
        } else if (strcmp(pArgs[i], "-theta") == 0) {
            configurations.tFilename = pArgs[i + 1];
        } else if (strcmp(pArgs[i], "-v") == 0) {
            configurations.vFilename = pArgs[i + 1];
        } else if (strcmp(pArgs[i], "-w") == 0) {
            configurations.wFilename = pArgs[i + 1];
        } else if (strcmp(pArgs[i], "-cov") == 0) {
            configurations.stateCovarianceModifier = atof(pArgs[i + 1]);
        }
    }

    return configurations;
}


std::vector<DataItem> readDataFile(const std::string& dataFileName) {
    std::vector<DataItem> data;

    try {
        std::ifstream fin;
        fin.clear();
        fin.open(dataFileName.c_str());

        std::string dummyString;
        int dummy;
        char delimeter;
        DataItem temp;

        fin >> dummyString;

        temp = readDataLine(fin);

        while (fin.good()) {
            data.push_back(temp);

            temp = readDataLine(fin);
        }

        fin.close();
    } catch (std::exception& e) {
        throw e;
    }

    return data;
}


DataItem readDataLine(std::ifstream& fin) {
    DataItem dataItem;
    dataItem.measurements.resize(NUM_MEASUREMENTS);
    dataItem.variances.resize(NUM_MEASUREMENTS);

    try {
        int dummy;
        char delimeter;

        dataItem.measurements.setZero();
        dataItem.variances.setConstant(NUM_MEASUREMENTS, 1, ARBITRARY_VARIANCE);

        fin >>
            dummy >> delimeter >>
            dataItem.measurements(ODOM_X) >> delimeter >> // odometer.x
            dataItem.measurements(ODOM_Y) >> delimeter >> // odometer.y
            dataItem.measurements(ODOM_T) >> delimeter >> // odometer.\theta
            dataItem.measurements(IMU_T) >> delimeter >>  // IMU.\theta
            dataItem.variances(IMU_T) >> delimeter >>     // IMU.\theta covariance
            dataItem.measurements(GPS_X) >> delimeter >>  // GPS.x
            dataItem.measurements(GPS_Y) >> delimeter >>  // GPS.y
            dataItem.variances(GPS_X) >> delimeter >>     // GPS.x covariance
            dataItem.variances(GPS_Y);                    // GPS.y covariance

            dataItem.measurements(IMU_T) += 0.172; // calibration - just matches things to start
    } catch (std::exception& e) {
        throw e;
    }

    return dataItem;
}


std::vector<KalmanFilter::KalmanState> applyKalmanFilterToData(
    Configurations& pConfigurations,
    std::vector<DataItem>& pData) {

    std::vector<KalmanFilter::KalmanState> results;

    try {
        KalmanFilter kalmanFilter(STATE_DIMENSIONS, MEASUREMENT_DIMENSIONS);

        Eigen::VectorXd tempMeasurements(NUM_MEASUREMENTS);
        tempMeasurements = pData[0].measurements;

        Eigen::VectorXd Z(MEASUREMENT_DIMENSIONS);
        Z = reduceDataToInputMeasurements(
            pData[0].measurements,
            MEASUREMENT_DIMENSIONS,
            {
                {GPS_X, X_ESTIMATE},
                {GPS_Y, Y_ESTIMATE},
                {IMU_T, T_ESTIMATE}
            }
        );

        Eigen::MatrixXd A(STATE_DIMENSIONS, STATE_DIMENSIONS);
        A = computeUpdatedNaturalModelMatrix(
                STATE_DIMENSIONS,
                DELTA_T,
                tempMeasurements
        );

        Eigen::MatrixXd B(STATE_DIMENSIONS, STATE_DIMENSIONS);
        B.setIdentity(STATE_DIMENSIONS, STATE_DIMENSIONS);

        Eigen::VectorXd u(STATE_DIMENSIONS);
        u.setZero();

        Eigen::MatrixXd H(MEASUREMENT_DIMENSIONS, STATE_DIMENSIONS);
        H.setIdentity(MEASUREMENT_DIMENSIONS, STATE_DIMENSIONS);

        Eigen::MatrixXd R(MEASUREMENT_DIMENSIONS, MEASUREMENT_DIMENSIONS);
        R = computeUpdatedObservationNoiseCovariance(MEASUREMENT_DIMENSIONS, 0.1, 0.1, 0.01);

        Eigen::MatrixXd Q(STATE_DIMENSIONS, STATE_DIMENSIONS);
        Q = computeUpdatedStateNoiseCovariance(STATE_DIMENSIONS, 0.00001, 0.00001);
        Q *= pConfigurations.stateCovarianceModifier;


        KalmanFilter::KalmanState previousState;
        previousState.state.resize(STATE_DIMENSIONS, 1);
        previousState.state.setZero();
        previousState.errorCovariance.resize(STATE_DIMENSIONS, STATE_DIMENSIONS);
        previousState.errorCovariance.setIdentity();
        previousState.errorCovariance *= 0.01;


        kalmanFilter.setNaturalModel(A);
        kalmanFilter.setControlModel(B);
        kalmanFilter.setTransitionModel(H);
        kalmanFilter.setStateNoiseCovariance(Q);
        kalmanFilter.setMeasurementNoiseCovariance(R);

        for (int i = 0; i < pData.size(); i++) {

            A = computeUpdatedNaturalModelMatrix(
                    STATE_DIMENSIONS,
                    DELTA_T,
                    previousState.state
            );
            R = computeUpdatedObservationNoiseCovariance(
                    MEASUREMENT_DIMENSIONS,
                    pData[i].variances(GPS_X),
                    pData[i].variances(GPS_Y),
                    pData[i].variances(IMU_T)
            );
            Z = reduceDataToInputMeasurements(
                pData[i].measurements,
                MEASUREMENT_DIMENSIONS,
                {
                    {GPS_X, X_ESTIMATE},
                    {GPS_Y, Y_ESTIMATE},
                    {IMU_T, T_ESTIMATE}
                }
            );

            kalmanFilter.setNaturalModel(A);
            kalmanFilter.setMeasurementNoiseCovariance(R);
            previousState = kalmanFilter.KalmanFilterIteration(previousState, Z, u);

            results.push_back(previousState);
        }

    } catch (std::exception& e) {
        throw e;
    }

    return results;
}


Eigen::MatrixXd computeUpdatedNaturalModelMatrix(
    int pStateDimensionality,
    double pTimeStepMagnitude,
    Eigen::VectorXd& pPreviousState) {

    double previousAngle = pPreviousState(T_ESTIMATE);
    double previousAngularVelocity = pPreviousState(W_ESTIMATE);

    Eigen::MatrixXd A(pStateDimensionality, pStateDimensionality);
    A << 1, 0, pTimeStepMagnitude * cos(previousAngle), 0,                  0,
         0, 1, pTimeStepMagnitude * sin(previousAngle), 0,                  0,
         0, 0,                                       1, 0,                  0,
         0, 0,                                       0, 1, pTimeStepMagnitude,
         0, 0,                                       0, 0,                  1;

    return A;
}


Eigen::MatrixXd computeUpdatedStateNoiseCovariance(
    int pStateDimensionality,
    double pXPosCovariance,
    double pYPosCovariance) {

    Eigen::MatrixXd Q(pStateDimensionality, pStateDimensionality);
    Q << pXPosCovariance,               0,     0,     0,     0,
                       0, pYPosCovariance,     0,     0,     0,
                       0,               0, 0.001,     0,     0,
                       0,               0,     0, 0.001,     0,
                       0,               0,     0,     0, 0.001;
    return Q;
}


Eigen::MatrixXd computeUpdatedObservationNoiseCovariance(
    int pObservationDimensionality,
    double pXPosCovariance, 
    double pYPosCovariance,
    double pTPosCovariance) {

    Eigen::MatrixXd R(pObservationDimensionality, pObservationDimensionality);
    R << pXPosCovariance,               0,    0,               0,    0,
                       0, pYPosCovariance,    0,               0,    0,
                       0,               0, 0.01,               0,    0,
                       0,               0,    0, pTPosCovariance,    0,
                       0,               0,    0,               0, 0.01;

    return R;
}


Eigen::VectorXd reduceDataToInputMeasurements(
    Eigen::VectorXd& pMeasurementData,
    int pNewVectorSize,
    const std::vector<std::pair<int, int>>& pDataIndices) {

    Eigen::VectorXd newMeasurementVector(pNewVectorSize);
    newMeasurementVector.setZero();

    for (std::pair<int, int> indices : pDataIndices) {
        newMeasurementVector(indices.second) = pMeasurementData(indices.first);
    }

    return newMeasurementVector;
}


void writeDataToFile(
    std::string pFileName,
    std::vector<DataItem>& pInputData,
    const std::vector<std::string>& pDataColumnLabels,
    const std::vector<int>& pDataIndices,  
    std::vector<KalmanFilter::KalmanState>& pKalmanOutputData,
    const std::vector<std::string>& pOutputColumnLabels,
    const std::vector<int>& pOutputIndices) {

    try {
        std::ofstream fout;

        fout.clear();
        fout.open(pFileName + ".txt");

        std::string fileComment = "# ";
        fileComment += pFileName;
        fileComment += "data";
        fout << fileComment << std::endl;

        fileComment = "# time\t";
        for (std::string label : pDataColumnLabels) {
            fileComment += label;
            fileComment += "\t";
        }
        for (std::string label : pOutputColumnLabels) {
            fileComment += label;
            fileComment += "\t";    
        }
        fout << fileComment << std::endl;

        for (int i = 0; i < pKalmanOutputData.size(); i++) {
            fout << std::setw(5);
            fout << i << '\t';

            for (int index : pDataIndices) {
                fout << std::setw(14);
                fout << pInputData[i].measurements(index) << "\t";
            }
            for (int index : pOutputIndices) {
                fout << std::setw(14);
                fout << pKalmanOutputData[i].state(index) << "\t";
            }

            fout << std::endl;
        }

        fout.close();

    } catch (std::exception& e) {
        throw e;
    }

}


void testSimpleKalmanFilter() {
        Eigen::MatrixXd A(1,1); A << 1;
        Eigen::MatrixXd B(1,1); B << 1;
        Eigen::MatrixXd H(1,1); H << 1;
        KalmanFilter::KalmanState state;
            state.state.resize(1);
            state.state << 0.0;
            state.errorCovariance.resize(1,1);
            state.errorCovariance << 1.0;
        Eigen::VectorXd Z(1); Z << 0.390;
        Eigen::VectorXd u(1); u << 0.0;
        Eigen::MatrixXd Q(1,1); Q << 0.0;
        Eigen::MatrixXd R(1,1); R << 0.1;

        KalmanFilter kalmanFilter(1, 1);
        kalmanFilter.setNaturalModel(A);
        kalmanFilter.setControlModel(B);
        kalmanFilter.setTransitionModel(H);
        kalmanFilter.setStateNoiseCovariance(Q);
        kalmanFilter.setMeasurementNoiseCovariance(R);

        Z << 0.390;
        state = kalmanFilter.KalmanFilterIteration(state, Z, u);
        std::cout << "state\n" << state.state << std::endl <<
                  "P\n" << state.errorCovariance << std::endl << std::endl;

                          Z << 0.5;
        state = kalmanFilter.KalmanFilterIteration(state, Z, u);
        std::cout << "state\n" << state.state << std::endl <<
                  "P\n" << state.errorCovariance << std::endl << std::endl;

                          Z << 0.480;
        state = kalmanFilter.KalmanFilterIteration(state, Z, u);
        std::cout << "state\n" << state.state << std::endl <<
                  "P\n" << state.errorCovariance << std::endl << std::endl;
}

#endif //__KALMANFILTER_MAIN_CPP__
