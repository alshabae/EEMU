cmake_minimum_required(VERSION 3.20)
project(cpp_ops)

set(CMAKE_CXX_STANDARD 14)
set(PYBIND11_PYTHON_VERSION 3.11)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -fPIC -fopenmp")

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(${PROJECT_NAME} binding.cpp)

target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_include_directories(${PROJECT_NAME} PRIVATE ${PYBIND11_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)