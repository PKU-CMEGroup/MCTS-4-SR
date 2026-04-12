// include/imcts/core/dataset.hpp
#pragma once
#include <vector>
#include <Eigen/Core>
#include "types.hpp"

namespace imcts {

class Dataset {
public:
    Eigen::MatrixXd X;  // (n_samples x n_vars), column-major
    Eigen::VectorXd y;  // (n_samples)

    Dataset() = default;

    // Construct from column vectors (each inner vector is one variable)
    Dataset(const std::vector<std::vector<float>>& x_cols,
            const std::vector<float>& y_vec)
    {
        int n_samples = static_cast<int>(y_vec.size());
        int n_vars    = static_cast<int>(x_cols.size());

        X.resize(n_samples, n_vars);
        for (int v = 0; v < n_vars; ++v) {
            for (int s = 0; s < n_samples; ++s) {
                X(s, v) = static_cast<double>(x_cols[v][s]);
            }
        }

        y.resize(n_samples);
        for (int s = 0; s < n_samples; ++s) {
            y(s) = static_cast<double>(y_vec[s]);
        }
    }

    [[nodiscard]] int n_samples() const { return static_cast<int>(X.rows()); }
    [[nodiscard]] int n_vars()    const { return static_cast<int>(X.cols()); }
};

} // namespace imcts
