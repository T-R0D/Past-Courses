#ifndef __TARGET_FIELD_CPP__
#define __TARGET_FIELD_CPP__ 1

#include "target_field.hpp"

#include <list>
#include <exception>

#include <Eigen/Dense>


TargetField::TargetField(
  const double p_x_size,
  const double p_y_size,
  const int p_num_cells
) : x_size_(p_x_size), y_size_(p_y_size), num_cells_(p_num_cells) {}

TargetField::~TargetField() {}

TargetField&
TargetField::add_target(const TargetField::Target& p_target) {
  targets_.push_back(p_target);
  return *this; // for call chaining
}

std::vector<TargetField::Target>
TargetField::get_targets() const {
  return targets_;
}

Eigen::MatrixXd
TargetField::as_cells() const {
  Eigen::MatrixXd cells(num_cells_, num_cells_);
  cells.setZero();

  return cells;
};

  std::pair<double, double>
  TargetField::coordinates_of_cell(
    const int p_cell_x,
    const int p_cell_y
  ) const {
    return {0, 0};
  }

/**
 * Inner Class things
 */
TargetField::Target::Target(
    const double p_value,
    const std::pair<double, double> p_coordinates
) : value_(p_value), coordinates_(p_coordinates) {}

TargetField::Target::~Target() {};

double
TargetField::Target::value() const {
  return value_;
}

std::pair<double, double>
TargetField::Target::coordinates() const {
  return coordinates_;
}

#endif //__TARGET_FIELD_CPP__