#pragma once
#include <random>

#include <Eigen/Dense>

inline std::mt19937& global_rng() {
    static std::mt19937 gen(42);
    return gen;
}

inline Eigen::VectorXi argmax_rows(const Eigen::MatrixXd& M) {
    Eigen::VectorXi out(M.rows());
    for (int i = 0; i < M.rows(); ++i) {
        M.row(i).maxCoeff(&out(i));
    }
    return out;
}

inline Eigen::MatrixXd softmax(const Eigen::MatrixXd& X) {
    // subtract max per row for stability
    Eigen::VectorXd rowMax = X.rowwise().maxCoeff();
    Eigen::MatrixXd Z = X;
    Z = Z.array().colwise() - rowMax.array();   // correct broadcasting

    Eigen::MatrixXd expZ = Z.array().exp();
    Eigen::VectorXd sumExp = expZ.rowwise().sum();

    auto probs = expZ.array().colwise() / sumExp.array();
    return probs;
}

inline double accuracy(const Eigen::VectorXi& y_true,
                       const Eigen::VectorXi& y_pred)
{
    int correct = 0;
    for (int i = 0; i < y_true.size(); ++i) {
        if (y_true(i) == y_pred(i)) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / y_true.size();
}

