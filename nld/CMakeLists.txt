
# Create the nld interface library
add_library(nld INTERFACE)

# target_include_directories(nld INTERFACE ${CMAKE_INSTALL_INCLUDEDIR})
# Add the include paths to the nld target
target_include_directories(nld
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
