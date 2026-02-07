#pragma once

#include <Eigen/dense>
#include <cassert>

class CrossEntropyLoss {
public:
  static double value(const Eigen::MatrixXd& y_true, 
                      const Eigen::MatrixXd& y_pred) 
  {
    assert(y_true.rows() == y_pred.rows());
    assert(y_true.cols() == y_pred.cols());

    const double eps {1e-9};
    std::size_t N { y_true.rows() };

    double loss { -(y_true.array() *(y_pred.array() + eps).log()).sum() };

    return loss / static_cast<double>(N);
  }

  static Eigen::MatrixXd gradient(const Eigen::MatrixXd& y_true,
                                  const Eigen::MatrixXd& y_pred)
  {
    assert(y_true.rows() == y_true.rows());
    assert(y_true.cols() == y_pred.cols());

    std::size_t N { y_true.rows() };
    return (y_pred - y_true) / static_cast<double>(N);
  }
};
