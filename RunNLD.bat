@echo off

if not exist build (
  mkdir build
)

cd build

cmake.exe -A x64 -T v141 .. -DCMAKE_PREFIX_PATH=C:/packages

start NonlinearDynamic.sln

cd ..