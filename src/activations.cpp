#include "nn/activations.hpp"
#include <cassert>

Eigen::MatrixXd ReLU::forward(const Eigen::MatrixXd& X) {
  Z_cache = X;
  return X.cwiseMax(0.0);
}

Eigen::MatrixXd ReLU::backward(const Eigen::MatrixXd& dY) {
  assert(dY.rows() == Z_cache.rows());
  assert(dY.cols() == Z_cache.cols()); 

  Eigen::MatrixXd dX = dY;
  // derivative is 1 if x > 0, else 0 (we use boolean and static it to double)
  dX.array() *= (Z_cache.array() > 0.0).cast<double>();
  return dX;
}

Eigen::MatrixXd Softmax::forward(const Eigen::MatrixXd& X) {
  // for stability
  Eigen::MatrixXd Z { X.rowwise() - X.rowwise().maxCoeff() };

  Eigen::MatrixXd expZ { Z.array().expwise() };
  Eigen::MatrixXd sumExp { Z.rowwise().sum() };

  probs = expZ.array().colwise() / sumExp.array();

  return probs;
}

Eigen::MatrixXd Softmax::backward(const Eigen::MatrixXd dY) { return dY; }
