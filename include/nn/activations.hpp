#pragma once 

#include <Eigen/Dense>
#include "nn/layer.hpp"

class ReLU : public Layer {
public:
  Eigen::MatrixXd forward(const Eigen::MatrixXd& X) override;
  Eigen::MatrixXd backward(const Eigen::MatrixXd& dY) override;
  void update(double lr) override {}

private:
  Eigen::MatrixXd Z_cache;
};

class Softmax : public Layer {
public:
  Eigen::MatrixXd forward(const Eigen::MatrixXd& X) override;
  Eigen::MatrixXd backward(const Eigen::MatrixXd& dY) override;
  void update(double lr) override {};

private:
  Eigen::MatrixXd probs;
};

