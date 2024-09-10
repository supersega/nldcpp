file(GLOB_RECURSE CPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
pybind11_add_module(nldpy ${CPP_FILES})

target_link_libraries(nldpy LINK_PRIVATE autodiff::autodiff Eigen3::Eigen)

target_include_directories(nldpy PUBLIC ${PROJECT_SOURCE_DIR})

find_package(Boost REQUIRED COMPONENTS Hana)
target_include_directories(nldpy PUBLIC ${Boost_INCLUDE_DIRS})

# configure_file(setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

# message(STATUS "Python site-packages directory: ${Python_SITELIB}")
#
# add_custom_target(nld-setuptools ALL
#     DEPENDS nldpy
#     COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml ${CMAKE_CURRENT_BINARY_DIR}
#     COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nldpy> ${Python_SITELIB} # Copy the .so file to the site-packages directory
#     COMMAND pip install . --force-reinstall
#     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
