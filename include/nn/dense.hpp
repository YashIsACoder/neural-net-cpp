#pragma once

#include "layer.hpp"
#include <Eigen/Dense>

class Dense : public Layer {
public:
  Dense(std::size_t input_dim, std::size_t output_dim);

  Eigen::MatrixXd forward(const Eigen::MatrixXd& X) override;
  Eigen::MatrixXd backward(const Eigen::MatrixXd& Y) override;
  void update(double lr) override;
  
private:
  // parameter
  Eigen::MatrixXd W; // (input_dim, output_dim) Weights
  Eigen::MatrixXd b; // (output_dim) bias
  
  Eigen::MatrixXd X_cache;

  // gradients
  Eigen::MatrixXd dW;
  Eigen::MatrixXd db; 
}
