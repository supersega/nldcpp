#!bin/zsh

cmake=/usr/local/bin/cmake
clang=/Volumes/Data/clang/clang+llvm-12.0.0-x86_64-apple-darwin/bin/clang
clangpp=/Volumes/Data/clang/clang+llvm-12.0.0-x86_64-apple-darwin/bin/clang++
vcpkg=/Volumes/Data/dev/vcpkg/scripts/buildsystems/vcpkg.cmake
build_dir=build
build_type=ReleaseWithDebInfo
enable_asan=FALSE

$cmake \
    --no-warn-unused-cli \
    -DCMAKE_BUILD_TYPE:STRING=$build_type \
    -DCMAKE_TOOLCHAIN_FILE:STRING=$vcpkg \
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
    -DCMAKE_C_COMPILER:FILEPATH=$clang \
    -DCMAKE_CXX_COMPILER:FILEPATH=$clangpp \
    -DENABLE_ASAN:BOOL=$enable_asan \
    -B $build_dir \
    -G "Unix Makefiles"

$cmake --build $build_dir --config $build_type --target all

