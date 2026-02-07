#pragma once

#include <Eigen/dense>
#include <cassert>

class CrossEntropyLoss {
public:
  static double value(const Eigen::MatrixXd& y,
                      const Eigen::MatrixXd& probs) {
      const double eps = 1e-12;
      Eigen::MatrixXd clipped = probs.array().max(eps).min(1.0 - eps);
      return -(y.array() * clipped.array().log()).sum() / y.rows();
  }

  static Eigen::MatrixXd gradient(const Eigen::MatrixXd& y,
                                  const Eigen::MatrixXd& probs) {
    return probs - y;  
  }
};
