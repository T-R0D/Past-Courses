#ifndef __KALMANFILTER_HPP__
#define __KALMANFILTER_HPP__

#include <Eigen/Dense>
#include <utility>


class KalmanFilter {
 public:
 	struct KalmanState {
 		Eigen::VectorXd state;
 		Eigen::MatrixXd errorCovariance;
 	};


 	KalmanFilter(
    const int pStateDimensionality,
    const int pMeasurementDimensionality);

  void setNaturalModel(const Eigen::MatrixXd& pNewModel);
  void setControlModel(const Eigen::MatrixXd& pNewModel);
  void setTransitionModel(const Eigen::MatrixXd& pNewModel);
  void setStateNoiseCovariance(const Eigen::MatrixXd& pNewCovariance);
  void setMeasurementNoiseCovariance(const Eigen::MatrixXd& pNewCovariance);

	KalmanState KalmanFilterIteration(
		const KalmanState& pPreviousState,
		const Eigen::MatrixXd& pMeasurementVector,
		const Eigen::VectorXd& pControlVector);

 	Eigen::MatrixXd computeStatePrediction(
 		const Eigen::VectorXd& pPreviousState,
 		const Eigen::VectorXd& pControlVector);

 	Eigen::MatrixXd computeObservationPrediction(
 		const Eigen::VectorXd& pStatePrediction);

 	Eigen::MatrixXd computeErrorCovariancePrediction(
 		const Eigen::MatrixXd& pPreviousPredictionCovariance);

 	Eigen::MatrixXd computeKalmanGainFactor(
 		const Eigen::MatrixXd& pPredictionCovarianceEstimate);

 	Eigen::MatrixXd computeStateEstimate(
 		const Eigen::VectorXd& pStatePrediction,
 		const Eigen::MatrixXd& pKalmanGainFactor,
 		const Eigen::VectorXd& pMeasurementVector,
 		const Eigen::VectorXd& pObservationPrediction);

 	Eigen::MatrixXd computeErrorCovariance(
 		const Eigen::MatrixXd& pPredictionCovarianceEstimate,
 		const Eigen::MatrixXd& pKalmanGainFactor);

 private:
  int mStateDimensionality;       // aka n
  int mMeasurementDimensionality; // aka m
  Eigen::MatrixXd mNaturalModel;  // aka A; n x n; describes how the system
                                  // evolves naturally, i.e. without controls
                                  // or noise
  Eigen::MatrixXd mControlModel;  // aka B; n x n; describes how controls alter
                                  // the system
  Eigen::MatrixXd mTransitionModel; // aka H; m x n; describes how to map from
                                    // a state to an observation
  Eigen::VectorXd mStateNoise;    // aka \epsilon; n x 1; has covariance Q
  Eigen::MatrixXd mStateNoiseCovariance;  // aka Q;
  Eigen::VectorXd mMeasurementNoise; // aka \sigma; m x 1; has covariance R
  Eigen::MatrixXd mMeasurementNoiseCovariance;  // aka R; m x m
};


#endif //__KALMANFILTER_HPP__