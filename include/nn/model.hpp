#include <Eigen/Dense>
#include <memory>
#include <vector>
#include "nn/layer.hpp"

class NeuralNetwork {
public:
  void add(std::unique_ptr<Layer> layer);

  void fit(const Eigen::MatrixXd& X, 
           const Eigen::MatrixXd& dY,
           std::size_t epochs,
           std::size_t batch_size,
           double lr);

  Eigen::MatrixXd predict_proba(const Eigen::MatrixXd& X);

  Eigen::VectorXi predict(const Eigen::MatrixXd& X);

  double score(const Eigen::MatrixXd& X, 
               const Eigen::MatrixXd& y);

private:
  std::vector<std::unique_ptr<Layer>> layers;
};
