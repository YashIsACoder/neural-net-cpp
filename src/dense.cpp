#include "nn/dense.hpp"
#include <random>
#include <cassert>
#include <iostream>
#include "nn/utils.hpp"

Dense::Dense(std::size_t input_dim, std::size_t output_dim)
  : W(input_dim, output_dim),
    b(output_dim),
    dW(input_dim, output_dim),
    db(output_dim)
{
    // He initialization (best for ReLU)
    double stddev = std::sqrt(2.0 / static_cast<double>(input_dim));

    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, stddev);

    for (int i = 0; i < W.rows(); ++i)
        for (int j = 0; j < W.cols(); ++j)
            W(i, j) = dist(global_rng());

    b.setZero();
}

Eigen::MatrixXd Dense::forward(const Eigen::MatrixXd& X) {
  // need to cache for backprop later
  X_cache = X; 

  // Z = W*x + b 
  Eigen::MatrixXd Z { X * W };
  Z.rowwise() += b.transpose();

  return Z;
}


Eigen::MatrixXd Dense::backward(const Eigen::MatrixXd& dY) {
    int N = X_cache.rows();

    dW = X_cache.transpose() * dY / N;
    db = dY.colwise().mean();

    return dY * W.transpose();
}

void Dense::update(double lr) {
    // std::cout << "W norm before: " << W.norm() << "\n";
    W -= lr * dW;
    b -= lr * db;
    // std::cout << "W norm after:  " << W.norm() << "\n";
}
