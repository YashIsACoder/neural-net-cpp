cmake_minimum_required(VERSION 3.14)
project(SimpleMLP LANGUAGES CXX)

# ----------------------------
# C++ standard
# ----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------------------------
# Global optimization flags
# ----------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

# ----------------------------
# Try to enable OpenMP
# ----------------------------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found — enabling multi-core acceleration")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

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

if(OpenMP_CXX_FOUND)
    target_link_libraries(nn PUBLIC OpenMP::OpenMP_CXX)
endif()

# ----------------------------
# Executable: train_mnist
# ----------------------------
add_executable(train_mnist
    examples/train_mnist.cpp
)

target_link_libraries(train_mnist
    PRIVATE nn
)

