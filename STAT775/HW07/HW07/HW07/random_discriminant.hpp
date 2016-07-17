#ifndef _RANDOM_DISCRIMINANT_
#define _RANDOM_DISCRIMINANT_

#include <math.h>

#include <Eigen/Dense>

struct DiscriminantModel {
  unsigned dim;
  Eigen::VectorXd projection_line;
  double decision_boundary;
};

void
InitDiscriminantModel(
  DiscriminantModel* self,
  unsigned dimensionality,
  const Eigen::MatrixXd* class_0_data,
  const Eigen::MatrixXd* class_1_data) {
/**
 *
 */
  self->dim = dimensionality;

  // get a random projection line (unit vector for easy projections)
  self->projection_line.resize(self->dim);
  unsigned i;
  for (i = 0; i < self->dim; ++i) {
    self->projection_line(i) = (double) (rand() % 100);
  }
  self->projection_line.normalize();

  Eigen::VectorXd class_0_projections = *class_0_data * self->projection_line;
  Eigen::VectorXd class_1_projections = *class_1_data * self->projection_line;

  double mu_0 = class_0_projections.mean();
  double mu_1 = class_1_projections.mean();
  double sigma_0 = 0.0;
  double sigma_1 = 0.0;
  for (i = 0; i < class_0_projections.size(); ++i) {
    sigma_0 += (class_0_projections(i) - mu_0) *
               (class_0_projections(i) - mu_0);
    sigma_1 += (class_1_projections(i) - mu_1) *
               (class_1_projections(i) - mu_1);
  }
  sigma_0 /= class_0_projections.size();
  sigma_1 /= class_1_projections.size();

  self->decision_boundary = 0.5 * (mu_1 - mu_0); // TODO: do actual boundary!!!!!!!!!!!
}


Eigen::VectorXi
GetDiscriminantPredictions(
  const DiscriminantModel* self,
  const Eigen::MatrixXd* observations) {
/**
 *
 */
  Eigen::VectorXd projections = *observations * self->projection_line;
std::cout << projections << std::endl;

  Eigen::VectorXi predictions(projections.size());
  unsigned i;
  for (i = 0; i < predictions.size(); ++i) {
    predictions(i) = (projections(i) >= self->decision_boundary) ? 0 : 1;
  }

  return predictions;
}

void
PrintDiscriminantModel(
  const DiscriminantModel* self,
  FILE* stream,
  const bool show_projection_line) {
/**
 *
 */
  printf("Dimensionality:    %u -> 1\n", self->dim);
  printf("Decision Boundary: %lf\n", self->decision_boundary);
  printf("Projection Line:\n");
  if (show_projection_line) {
    std::cout << self->projection_line << std::endl;
  }
}

#endif // define _RANDOM_DISCRIMINANT_