#pragma once

#include <Eigen/Dense>

struct MNIST {
    Eigen::MatrixXd X_train, y_train;
    Eigen::MatrixXd X_test, y_test;
};

MNIST load_mnist(const std::string& path);

