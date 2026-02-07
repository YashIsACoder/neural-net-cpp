#include <iostream>

#include "nn/model.hpp"
#include "nn/dense.hpp"
#include "nn/activations.hpp"
#include "nn/mnist.hpp"

int main() {
    try {
        // Load MNIST
        std::cout << "Loading MNIST..." << std::endl;
        MNIST mnist = load_mnist("data");

        std::cout << "Train samples: " << mnist.X_train.rows() << std::endl;
        std::cout << "Test samples:  " << mnist.X_test.rows() << std::endl;


        // Build MLP (MLPClassifier)
        NeuralNetwork model;

        model.add(std::make_unique<Dense>(784, 128));
        model.add(std::make_unique<ReLU>());

        model.add(std::make_unique<Dense>(128, 64));
        model.add(std::make_unique<ReLU>());

        model.add(std::make_unique<Dense>(64, 10));
        //model.add(std::make_unique<Softmax>());

        // Train
        std::cout << "Training..." << std::endl;
        model.fit(
            mnist.X_train,
            mnist.y_train,
            25,
            64,
            0.01
        );

        // Evaluate
        double acc = model.score(mnist.X_test, mnist.y_test);
        std::cout << "Test accuracy: " << acc * 100.0 << "%" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

