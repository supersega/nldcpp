# Configure the setup.py file
configure_file(setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
message(STATUS "NLD Current binary dir: ${CMAKE_CURRENT_BINARY_DIR}")

# Create a custom target to build the python package during build stage
add_custom_target(nld-setuptools ALL
    DEPENDS nldpy
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/nld ${CMAKE_CURRENT_BINARY_DIR}/nld
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nldpy> ${CMAKE_CURRENT_BINARY_DIR}/nld
    COMMAND pip install . --force-reinstall
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/nld/$<TARGET_FILE_NAME:nldpy>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
