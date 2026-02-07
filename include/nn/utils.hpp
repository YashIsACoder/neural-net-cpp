#pragma once

#include <Eigen/Dense>

inline Eigen::VectorXi argmax_rows(const Eigen::MatrixXd& M) {
    Eigen::VectorXi out(M.rows());
    for (int i = 0; i < M.rows(); ++i) {
        M.row(i).maxCoeff(&out(i));
    }
    return out;
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

