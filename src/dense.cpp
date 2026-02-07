#include "nn/dense.hpp"
#include <random>
#include <cassert>

Dense::Dense(std::size_t input_dim, std::size_t output_dim) 
  : W(input_dim, output_dim),
    b(output_dim),
    dW(input_dim, output_dim),
    db(output_dim)
{
  // Xavier init. to avoid vanishing gradient 
  double limit { std::sqrt(6 / (input_dim + output_dim)) };

  // random generation of weights
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<double> dist(-limit, limit);
  for (int i{}; i < W.rows; ++i)
    for (int j{}; j < W.cols(); ++j)
      W(i, j) = dist(gen);

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
  assert(X_cache.rows() == dY.rows());

  std::size_t batch_size { X_cache.rows() };

  // calculate grads 
  dW = (X_cache.transpose() * dY) / static_cast<double>(batch_size);
  db = dY.colwise().mean();

  // grad wrt input
  Eigen::MatrixXd dX = dY * W.transpose();
  return dX;
}

void Dense::update(double lr) {
  // save memory wt noalias 
  W.noalias() -= lr * dW;
  b.noalias() -= lr * db;
}
