#ifndef __KALMANFILTER_CPP__
#define __KALMANFILTER_CPP__

#include "KalmanFilter.hpp"
#include <iostream>

 KalmanFilter::KalmanFilter(
    const int pStateDimensionality,
    const int pMeasurementDimensionality) :
    mStateDimensionality(pStateDimensionality),
	mMeasurementDimensionality(pMeasurementDimensionality) {

  mNaturalModel.resize(mStateDimensionality, mStateDimensionality);
  mNaturalModel.setIdentity(mStateDimensionality, mStateDimensionality);

  mControlModel.resize(mStateDimensionality, mStateDimensionality);
  mControlModel.setIdentity(mStateDimensionality, mStateDimensionality);

  mTransitionModel.resize(mMeasurementDimensionality, mStateDimensionality);
  mTransitionModel.setIdentity(mMeasurementDimensionality, mStateDimensionality);

  mStateNoise.resize(mStateDimensionality);
  mStateNoise.setZero(mStateDimensionality);

  mStateNoiseCovariance.resize(mStateDimensionality, mStateDimensionality);
  mStateNoiseCovariance.setOnes(mStateDimensionality, mStateDimensionality);


  mMeasurementNoise.resize(mMeasurementDimensionality);
  mMeasurementNoise.setZero(mMeasurementDimensionality);

  mMeasurementNoiseCovariance.resize(mMeasurementDimensionality, mMeasurementDimensionality);
  mMeasurementNoiseCovariance.setOnes(mMeasurementDimensionality, mMeasurementDimensionality);
}




KalmanFilter::KalmanState KalmanFilter::KalmanFilterIteration(
	const KalmanState& pPreviousState,
	const Eigen::MatrixXd& pMeasurementVector,
	const Eigen::VectorXd& pControlVector) {

	Eigen::MatrixXd statePrediction =
		computeStatePrediction(pPreviousState.state, pControlVector);

	Eigen::MatrixXd observationPrediction =
		computeObservationPrediction(statePrediction);

	Eigen::MatrixXd predictionCovarianceEstimate =
		computeErrorCovariancePrediction(pPreviousState.errorCovariance);	

	Eigen::MatrixXd kalmanGainFactor =
		computeKalmanGainFactor(predictionCovarianceEstimate);

	KalmanState currentState = {
		computeStateEstimate(
			statePrediction,
			kalmanGainFactor,
			pMeasurementVector,
			observationPrediction
		),
		computeErrorCovariance(predictionCovarianceEstimate, kalmanGainFactor)
	};

	return currentState;
}


Eigen::MatrixXd KalmanFilter::computeStatePrediction(
 		const Eigen::VectorXd& pPreviousState,
 		const Eigen::VectorXd& pControlVector) {

	// X^hat = A * X(k-1) + B * u + \epsilon
	return mNaturalModel * pPreviousState +
	       mControlModel * pControlVector +
		   mStateNoise;
}


Eigen::MatrixXd KalmanFilter::computeObservationPrediction(
 		const Eigen::VectorXd& pStatePrediction) {

	// Y = H * X^hat + \sigma
	return mTransitionModel * pStatePrediction + mMeasurementNoise;
}


Eigen::MatrixXd KalmanFilter::computeErrorCovariancePrediction(
	const Eigen::MatrixXd& pPreviousPredictionCovariance) {

	// P^hat(t) = A * P^hat(t - 1) * A^t + Q
	return
	mNaturalModel * pPreviousPredictionCovariance * mNaturalModel.transpose() +
	mStateNoiseCovariance;
}


Eigen::MatrixXd KalmanFilter::computeKalmanGainFactor(
 	const Eigen::MatrixXd& pPredictionCovarianceEstimate) {

	// K = (P^hat * H) / (H * P^hat * H^t + R)
	Eigen::MatrixXd numerator =
		pPredictionCovarianceEstimate * mTransitionModel.transpose();
	Eigen::MatrixXd denominator =
		(mTransitionModel * pPredictionCovarianceEstimate * mTransitionModel.transpose()) +
		mMeasurementNoiseCovariance;

	return numerator * denominator.inverse(); // inverse works? make sure
}


Eigen::MatrixXd KalmanFilter::computeStateEstimate(
 		const Eigen::VectorXd& pStatePrediction,
 		const Eigen::MatrixXd& pKalmanGainFactor,
 		const Eigen::VectorXd& pMeasurementVector,
 		const Eigen::VectorXd& pObservationPrediction) {

	// X = X^hat + K(Z - Y)
	return pStatePrediction +
		pKalmanGainFactor * (pMeasurementVector - pObservationPrediction);
}


Eigen::MatrixXd KalmanFilter::computeErrorCovariance(
	const Eigen::MatrixXd& pPredictionCovarianceEstimate,
	const Eigen::MatrixXd& pKalmanGainFactor) {

	// P = P^hat - K * H * P^hat or P = (I - K*H) * P^hat
	return pPredictionCovarianceEstimate -
		pKalmanGainFactor * mTransitionModel * pPredictionCovarianceEstimate;
}


void KalmanFilter::setNaturalModel(const Eigen::MatrixXd& pNewModel) {
	assert(pNewModel.rows() == mStateDimensionality &&
	       pNewModel.cols() == mStateDimensionality);
	mNaturalModel = pNewModel;
}


void KalmanFilter::setControlModel(const Eigen::MatrixXd& pNewModel) {
	assert(pNewModel.rows() == mStateDimensionality &&
		   pNewModel.cols() == mStateDimensionality);
	mControlModel = pNewModel;
}


void KalmanFilter::setTransitionModel(const Eigen::MatrixXd& pNewModel) {

	mTransitionModel = pNewModel;
}


void KalmanFilter::setStateNoiseCovariance(const Eigen::MatrixXd& pNewCovariance) {
	mStateNoiseCovariance.resize(pNewCovariance.rows(), pNewCovariance.cols());
	mStateNoiseCovariance = pNewCovariance;
}


void KalmanFilter::setMeasurementNoiseCovariance(const Eigen::MatrixXd& pNewCovariance) {
	mMeasurementNoiseCovariance.resize(pNewCovariance.rows(), pNewCovariance.cols());
	mMeasurementNoiseCovariance = pNewCovariance;
}


#endif //__KALMANFILTER_CPP__