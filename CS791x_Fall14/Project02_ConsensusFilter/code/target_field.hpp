#ifndef __TARGET_FIELD_HPP__
#define __TARGET_FIELD_HPP__ 1

#include <list>
#include <vector>
#include <Eigen/Dense>

class TargetField {
 public:
  const double ignore = -666.666;

  class Target {
   public:
    Target(
      const double p_value,
      const std::pair<double, double> p_coordinates
    );
    
    ~Target();
    
    double value() const;
    
    std::pair<double, double> coordinates() const;

   private:
    double value_;
    std::pair<double, double> coordinates_;
  };

  TargetField(
    const double p_x_size,
    const double p_y_size,
    const int p_num_cells
  );

  ~TargetField();
  
  TargetField& add_target(const Target& p_target);
  
  std::vector<Target> get_targets() const;
  
  Eigen::MatrixXd as_cells() const;

  std::pair<double, double> coordinates_of_cell(
    const int p_cell_x,
    const int p_cell_y
  ) const;

 private:
  int num_cells_;
  double x_size_;
  double y_size_;
  std::vector<Target> targets_;
};

#endif //__TARGET_FIELD_HPP__