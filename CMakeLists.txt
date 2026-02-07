cmake_minimum_required(VERSION 3.14)
project(SimpleMLP LANGUAGES CXX)

# ----------------------------
# C++ standard
# ----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------------------------
# Find Eigen
# ----------------------------
find_package(Eigen3 REQUIRED NO_MODULE)

# ----------------------------
# Library: nn
# ----------------------------
add_library(nn
    src/dense.cpp
    src/activations.cpp
    src/model.cpp
    src/mnist.cpp
)

target_include_directories(nn
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(nn
    PUBLIC Eigen3::Eigen
)

# ----------------------------
# Executable: train_mnist
# ----------------------------
add_executable(train_mnist
    examples/train_mnist.cpp
)

target_link_libraries(train_mnist
    PRIVATE nn
)


