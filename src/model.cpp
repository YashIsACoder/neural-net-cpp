#include "nn/model.hpp"
#include "nn/loss.hpp"
#include "nn/utils.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>

void NeuralNetwork::add(std::unique_ptr<Layer> layer) { layers.push_back(std::move(layer)); }

void NeuralNetwork::fit(const Eigen::MatrixXd& X, 
                        const Eigen::MatrixXd& y,
                        std::size_t epochs,
                        std::size_t batch_size,
                        double lr) 
{
  std::size_t N = X.rows();
  std::vector<std::size_t> indices(N);
  std::iota(indices.begin(), indices.end(), 0); // init vec
  
  std::mt19937 gen(std::random_device{}());

  for (std::size_t epoch{}; epoch < epochs; ++epoch) {
    std::shuffle(indices.begin(), indices.end(), gen);

    double epoch_loss{};

    for (std::size_t i{}; i < N; i += batch_size) {
      std::size_t end = std::min(i + batch_size, N);
      std::size_t bs = end - i;

      Eigen::MatrixXd X_batch(bs, X.cols());
      Eigen::MatrixXd y_batch(bs, y.cols());
      
      for (std::size_t j{}; j < bs; ++j) {
        X_batch.row(j) = X.row(indices[i + j]);
        y_batch.row(j) = y.row(indices[i + j]);
      }

      // forward
      Eigen::MatrixXd out = X_batch;
      for (auto& layer : layers)
        out = layer->forward(out);
      
      /*
      std::cout << "Output row 0: " << out.row(0) << "\n"; 
      std::cout << "Sum: " << out.row(0).sum() << "\n";
      std::cout << "Label row 0: " << y_batch.row(0) << "\n";
      */ 

      epoch_loss += CrossEntropyLoss::value(y_batch, out);

      // backward
      Eigen::MatrixXd grad = CrossEntropyLoss::gradient(y_batch, out);
      for (int l { static_cast<int>(layers.size()) - 1 }; l >= 0; --l)
        grad = layers[l]->backward(grad);

      // update
      for (auto& layer : layers)
        layer->update(lr);
    }

    epoch_loss /= static_cast<double>(N / batch_size);
    std::cout << "Epoch " << epoch + 1 << " | Loss : "
              << epoch_loss << std::endl;
  }
}

Eigen::MatrixXd NeuralNetwork::predict_proba(const Eigen::MatrixXd& X) {
  Eigen::MatrixXd out = X;
  for (auto& layer : layers)
    out = layer->forward(out);
  return out;
}

Eigen::VectorXi NeuralNetwork::predict(const Eigen::MatrixXd& X) {
  Eigen::MatrixXd probs { predict_proba(X) };
  return argmax_rows(probs);
}

double NeuralNetwork::score(const Eigen::MatrixXd& X, 
                            const Eigen::MatrixXd& y)
{
  Eigen::VectorXi y_pred = predict(X);
  Eigen::VectorXi y_true = argmax_rows(y);
  return accuracy(y_true, y_pred);
}
