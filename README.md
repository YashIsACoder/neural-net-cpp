# C++ MLP Neural Network Library

![C++](https://img.shields.io/badge/Language-C%2B%2B-blue) ![Eigen](https://img.shields.io/badge/Dependency-Eigen-lightgrey) ![License](https://img.shields.io/badge/License-MIT-green)

A **from-scratch C++ implementation of a Multilayer Perceptron (MLP) neural network**, inspired by scikit-learn’s `MLPClassifier`. Designed for **educational purposes, experimentation, and lightweight projects**, this library supports training on MNIST and other datasets with minimal dependencies.

---

## Features

* Fully-connected feedforward neural network (MLP)
* Multiple layers with arbitrary sizes
* Activation functions: ReLU, Sigmoid, Tanh, Softmax (for inference)
* Cross-entropy loss for classification
* He (Kaiming) weight initialization for ReLU networks
* Mini-batch stochastic gradient descent
* Compatible with Eigen for fast linear algebra
* Easy to extend for other optimizers or datasets
* Supports training and inference on MNIST with reproducible results

---

## Project Structure

```
cpp-mlp
├── CMakeLists.txt
├── data
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
│   ├── train-images-idx3-ubyte
│   └── train-labels-idx1-ubyte
├── examples
│   └── train_mnist.cpp
├── include
│   └── nn
│       ├── activations.hpp
│       ├── dense.hpp
│       ├── layer.hpp
│       ├── loss.hpp
│       ├── mnist.hpp
│       ├── model.hpp
│       └── utils.hpp
├── src
│   ├── activations.cpp
│   ├── dense.cpp
│   ├── mnist.cpp
│   └── model.cpp
└── train_mnist
```

---

## Dependencies

* **[Eigen](https://eigen.tuxfamily.org/)** — header-only linear algebra library (required)
* Standard C++17 compiler

> No heavy ML frameworks required — lightweight and portable.

---

## Build Instructions

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cpp-mlp.git
cd cpp-mlp
```

2. Configure with CMake:

```bash
mkdir build
cd build
cmake ..
```

3. Build the project:

```bash
cmake --build .
```

4. Run MNIST training example:

```bash
./train_mnist
```

---

## Usage Example

### Train on MNIST

```cpp
#include "nn/model.hpp"
#include "nn/mnist.hpp"
#include "nn/utils.hpp"

int main() {
    set_seed(42);  // reproducibility

    MNISTDataset mnist("data/");
    NeuralNetwork model;
    model.add_layer(Dense(784, 128));
    model.add_layer(ReLU());
    model.add_layer(Dense(128, 10));

    model.train(mnist.train_images, mnist.train_labels,
                epochs=25, batch_size=64, learning_rate=0.01);

    double acc = model.test(mnist.test_images, mnist.test_labels);
    std::cout << "Test accuracy: " << acc << "%" << std::endl;
}
```

---

### Predict New Samples

```cpp
Eigen::MatrixXd probs = model.predict_proba(X_new);
Eigen::VectorXi labels = model.predict(X_new); // argmax along classes
```

---

## Performance Notes

* Lightweight C++ implementation — suitable for small to medium datasets
* Training MNIST with 2-layer MLP (~128 hidden neurons) typically achieves **97%+ accuracy** in ~25 epochs
* Fully reproducible via global random seed

---

## License

MIT License — free to use, modify, and redistribute.

---

## Acknowledgements

* Inspired by scikit-learn’s `MLPClassifier`
* Eigen library for matrix operations
* MNIST dataset from [Yann LeCun](http://yann.lecun.com/exdb/mnist/)

