cmake_minimum_required(VERSION 3.0...3.28.3)

PROJECT(NonlinearDynamic VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Eigen3 REQUIRED)
find_package(GSL REQUIRED)

add_subdirectory(autodiff)
add_subdirectory(nld)

if (ENABLE_ASAN)
    message(STATUS "Enabling AddressSanitizer")
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif ()

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(examples)
    enable_testing()
    add_subdirectory(test)
endif ()
