cmake_minimum_required(VERSION 3.20)
project(MCTS_4_SR VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ---- Dependencies ----
find_package(OpenMP)

option(IMCTS_USE_SYSTEM_EIGEN "Prefer a system-installed Eigen3 before fetching a local copy" ON)
if(IMCTS_USE_SYSTEM_EIGEN)
    find_package(Eigen3 QUIET)
endif()
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found; fetching Eigen 3.4.0 for this build.")
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(eigen)
    if(TARGET eigen AND NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen ALIAS eigen)
    endif()
endif()
if(NOT TARGET Eigen3::Eigen)
    message(FATAL_ERROR "Eigen3 is required, but neither a system package nor the bundled fetch path succeeded.")
endif()

option(IMCTS_BUILD_PYTHON "Build Python bindings" ON)
if(IMCTS_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    option(IMCTS_USE_SYSTEM_PYBIND11 "Prefer a system-installed pybind11 before fetching a local copy" ON)
    if(IMCTS_USE_SYSTEM_PYBIND11)
        find_package(pybind11 CONFIG QUIET)
    endif()
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found; fetching pybind11 2.13.6 for this build.")
        FetchContent_Declare(
            pybind11
            URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    if(NOT TARGET pybind11::module)
        message(FATAL_ERROR "pybind11 is required for IMCTS_BUILD_PYTHON, but neither a system package nor the bundled fetch path succeeded.")
    endif()
endif()

if(BUILD_TESTING)
    include(CTest)
endif()

# ---- Core library ----
add_library(imcts_core STATIC
    source/core/exp_tree.cpp
    source/core/exp_queue.cpp
    source/core/bridge.cpp
    source/core/tree.cpp
    source/eval/interpreter.cpp
    source/eval/optimizer.cpp
    source/evaluator/evaluator.cpp
    source/gp/gp_manager.cpp
    source/mcts/mcts.cpp
    source/regressor.cpp
)
set_property(TARGET imcts_core PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(imcts_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(imcts_core PUBLIC Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(imcts_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---- Python module (optional) ----
if(IMCTS_BUILD_PYTHON)
    pybind11_add_module(imcts_py python/bindings.cpp)
    target_link_libraries(imcts_py PRIVATE imcts_core)
    install(TARGETS imcts_py
        LIBRARY DESTINATION imcts
        RUNTIME DESTINATION imcts
    )
    install(FILES
        imcts/__init__.py
        imcts/pretty.py
        DESTINATION imcts
    )
    install(FILES
        imcts/benchmarks/__init__.py
        imcts/benchmarks/__main__.py
        imcts/benchmarks/runner.py
        imcts/benchmarks/basic.yaml
        imcts/benchmarks/blackbox.yaml
        imcts/benchmarks/blackbox.json
        imcts/benchmarks/basic.json
        DESTINATION imcts/benchmarks
    )

    if(BUILD_TESTING)
        add_test(
            NAME python_smoke
            COMMAND
                ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/python/run_python_smoke.py
                $<TARGET_FILE_DIR:imcts_py>
        )
    endif()
endif()

# ---- Tests ----
if(BUILD_TESTING)
    add_subdirectory(test)
endif()
