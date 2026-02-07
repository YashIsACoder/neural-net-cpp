#pragma once

#include <Eigen/Dense>

inline Eigen::VectorXi argmax_rows(const Eigen::MatrixXd& M) {
    Eigen::VectorXi out(M.rows());
    for (int i = 0; i < M.rows(); ++i) {
        M.row(i).maxCoeff(&out(i));
    }
    return out;
}

// ----------------------------
// Numerically stable softmax
// ----------------------------
inline Eigen::MatrixXd softmax(const Eigen::MatrixXd& X) {
    // Step 1: subtract max per row for numerical stability
    Eigen::VectorXd rowMax = X.rowwise().maxCoeff();
    Eigen::MatrixXd Z = X;
    Z = Z.array().colwise() - rowMax.array();  // broadcast max along columns

    // Step 2: exponentiate
    Eigen::MatrixXd expZ = Z.array().exp();

    // Step 3: divide by row-wise sum
    Eigen::VectorXd sumExp = expZ.rowwise().sum();
    Eigen::MatrixXd probs = expZ.array().colwise() / sumExp.array();

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

