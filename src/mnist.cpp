#include "nn/mnist.hpp"
#include <fstream>
#include <vector>
#include <stdexcept>

static uint32_t read_int(std::ifstream& f) {
    unsigned char bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (uint32_t(bytes[0]) << 24) |
           (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8)  |
            uint32_t(bytes[3]);
}

static Eigen::MatrixXd read_images(const std::string& file) {
    std::ifstream f(file, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + file);

    int magic = read_int(f);
    int n = read_int(f);
    int rows = read_int(f);
    int cols = read_int(f);

    Eigen::MatrixXd X(n, rows * cols);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char pixel;
            f.read(reinterpret_cast<char*>(&pixel), 1);
            X(i, j) = pixel / 255.0;
        }
    }
    return X;
}

static Eigen::MatrixXd read_labels(const std::string& file, int num_classes) {
    std::ifstream f(file, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + file);

    int magic = read_int(f);
    int n = read_int(f);

    Eigen::MatrixXd y = Eigen::MatrixXd::Zero(n, num_classes);

    for (int i = 0; i < n; ++i) {
        unsigned char label;
        f.read(reinterpret_cast<char*>(&label), 1);
        y(i, label) = 1.0;
    }
    return y;
}

MNIST load_mnist(const std::string& path) {
    MNIST mnist;

    mnist.X_train = read_images(path + "/train-images-idx3-ubyte");
    mnist.y_train = read_labels(path + "/train-labels-idx1-ubyte", 10);

    mnist.X_test  = read_images(path + "/t10k-images-idx3-ubyte");
    mnist.y_test  = read_labels(path + "/t10k-labels-idx1-ubyte", 10);

    return mnist;
}

