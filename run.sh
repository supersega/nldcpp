/usr/local/bin/cmake --no-warn-unused-cli -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -DCMAKE_TOOLCHAIN_FILE:STRING=/Volumes/Data/dev/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_C_COMPILER:FILEPATH=/Volumes/Data/clang/clang+llvm-12.0.0-x86_64-apple-darwin/bin/clang -DCMAKE_CXX_COMPILER:FILEPATH=/Volumes/Data/clang/clang+llvm-12.0.0-x86_64-apple-darwin/bin/clang++ -S/Volumes/Data/dev/nldcpp -B/Volumes/Data/dev/nldcpp/build -G "Unix Makefiles"

/usr/local/bin/cmake --build /Volumes/Data/dev/nldcpp/build --config RelWithDebInfo --target all -j 10 --
