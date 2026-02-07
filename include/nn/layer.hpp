#pragma once 
#include <Eigen/Dense>

class Layer {
public:
  virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& X) = 0;
  virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& dY) = 0;
  virtual void update(double lr) = 0;
  virtual ~Layer() = default;
};
