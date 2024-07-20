find_package(Catch2 REQUIRED)

set(CMAKE_REQUIRED_LINK_OPTIONS "-arch;arm64")

file(GLOB_RECURSE TEST_SRC "*.test.cpp")

add_executable(nldtest ${TEST_SRC})

target_link_libraries(nldtest LINK_PRIVATE autodiff::autodiff Eigen3::Eigen)
target_link_libraries(nldtest PRIVATE Catch2::Catch2WithMain)
target_link_libraries(nldtest PRIVATE nld)
target_link_libraries(nldtest PRIVATE GSL::gsl GSL::gslcblas)
target_include_directories(nldtest PUBLIC ${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})

include(Catch)

catch_discover_tests(nldtest)
