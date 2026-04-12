// python/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "imcts/regressor.hpp"

namespace py = pybind11;
using namespace imcts;

PYBIND11_MODULE(imcts_py, m) {
    m.doc() = "iMCTS C++ symbolic regression (pybind11 interface)";

    py::class_<RegressorConfig>(m, "RegressorConfig")
        .def(py::init<>())
        .def_readwrite("ops",               &RegressorConfig::ops)
        .def_readwrite("max_depth",         &RegressorConfig::max_depth)
        .def_readwrite("K",                 &RegressorConfig::K)
        .def_readwrite("c",                 &RegressorConfig::c)
        .def_readwrite("gamma",             &RegressorConfig::gamma)
        .def_readwrite("gp_rate",           &RegressorConfig::gp_rate)
        .def_readwrite("mutation_rate",     &RegressorConfig::mutation_rate)
        .def_readwrite("exploration_rate",  &RegressorConfig::exploration_rate)
        .def_readwrite("max_unary",         &RegressorConfig::max_unary)
        .def_readwrite("max_constants",     &RegressorConfig::max_constants)
        .def_readwrite("lm_iterations",     &RegressorConfig::lm_iterations)
        .def_readwrite("max_evals",         &RegressorConfig::max_evals)
        .def_readwrite("succ_error_tol",    &RegressorConfig::succ_error_tol);

    py::class_<FitResult>(m, "FitResult")
        .def_readonly("best_path",   &FitResult::best_path)
        .def_readonly("best_coefficients", &FitResult::best_coefficients)
        .def_readonly("expression",  &FitResult::expression)
        .def_readonly("best_reward", &FitResult::best_reward)
        .def_readonly("n_evals",     &FitResult::n_evals);

    py::class_<Regressor>(m, "Regressor")
        .def(py::init([](
            py::array_t<float> x_train,
            py::array_t<float> y_train,
            const RegressorConfig& cfg)
        {
            auto x_buf = x_train.unchecked<2>();
            auto y_buf = y_train.unchecked<1>();
            int n_vars    = static_cast<int>(x_train.shape(0));
            int n_samples = static_cast<int>(x_train.shape(1));

            std::vector<std::vector<float>> x_cols(n_vars, std::vector<float>(n_samples));
            for (int i = 0; i < n_vars; i++)
                for (int j = 0; j < n_samples; j++)
                    x_cols[i][j] = x_buf(i, j);

            std::vector<float> y(n_samples);
            for (int j = 0; j < n_samples; j++)
                y[j] = y_buf(j);

            return Regressor(std::move(x_cols), std::move(y), cfg);
        }),
        py::arg("x_train"), py::arg("y_train"),
        py::arg("cfg") = RegressorConfig{})
        .def("fit", [](Regressor& self, py::object seed) -> FitResult {
            if (seed.is_none()) return self.fit();
            return self.fit(seed.cast<uint64_t>());
        }, py::arg("seed") = py::none());
}
