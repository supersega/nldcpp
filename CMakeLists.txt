cmake_minimum_required(VERSION 3.0...3.28.3)

PROJECT(NonlinearDynamic VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Eigen3 REQUIRED)
find_package(GSL REQUIRED)

option(NLD_BUILD_TESTS "Build the tests." OFF)
option(NLD_BUILD_EXAMPLES "Build the examples." OFF)
option(NLD_BUILD_PYTHON "Build the python bindings." OFF)

message(STATUS "Building tests: ${NLD_BUILD_TESTS}")
message(STATUS "Building examples: ${NLD_BUILD_EXAMPLES}")
message(STATUS "Building python bindings: ${NLD_BUILD_PYTHON}")

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

if (NLD_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development)
    message(STATUS "Python executable found in: ${Python_EXECUTABLE}")
    find_package(pybind11 REQUIRED)
    option(AUTODIFF_BUILD_PYTHON "Enable the compilation of the python bindings." ON)
endif ()

# Never build autodiff tests and examples
option(AUTODIFF_BUILD_TESTS "Disable the compilation of the tests." OFF)
option(AUTODIFF_BUILD_EXAMPLES "Disable the compilation of the examples." OFF)

add_subdirectory(autodiff)

add_subdirectory(nld)

if (NLD_BUILD_PYTHON)
    add_subdirectory(python)
endif ()

if (NLD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (NLD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()

if (ENABLE_ASAN)
    message(STATUS "Enabling AddressSanitizer")
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif ()

