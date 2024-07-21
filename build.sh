#!/bin/zsh

RED='\033[0;31m'
NC='\033[0m'

function log_error() {
    echo "${RED}[ERROR]: $1${NC}"
}

function print_help() {
    echo "Build script to build the nld with clang and vcpkg"
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help              Print this help message"
    echo "  -m, --cmake_path        Path to cmake"
    echo "  -c, --clang_path        Path to clang"
    echo "  -p, --clangpp_path      Path to clang++"
    echo "  -v, --vcpkg_path        Path to vcpkg cmake file, mandatory option"
    echo "  -t, --build_type        Build type"
    echo "  -d, --build_dir         Build directory"
    echo "  -a, --enable_asan       Enable AddressSanitizer"
    echo "    , --build_tests       Build tests"
    echo "    , --build_examples    Build examples"
    echo "    , --build_python      Build python bindings"
    echo "    , --build_all         Build all"
    exit 1
}

opts=$(getopt --options hm:c:p:v:t:d:a --longoptions help,cmake_path:,clang_path:,clangpp_path:,vcpkg_path:,build_type:,build_dir:,enable_asan -- "$@") || print_help

while (($#))
do
    case $1 in
		-h|--help)         print_help;;
        -m|--cmake_path)   CMAKE_PATH=$2; shift;;
        -c|--clang_path)   CLANG_PATH=$2; shift;;
        -p|--clangpp_path) CLANGPP_PATH=$2; shift;;
        -v|--vcpkg_path)   VCPKG_PATH=$2; shift;;
        -t|--build_type)   BUILD_TYPE=$2; shift;;
        -a|--enable_asan)  ENABLE_ASAN=$2; shift;;
        --buld_tests)      BUILD_TESTS=TRUE;;
        --build_examples)  BUILD_EXAMPLES=TRUE;;
        --build_python)    BUILD_PYTHON=TRUE;;
        --build_all)       BUILD_TESTS=TRUE; BUILD_EXAMPLES=TRUE; BUILD_PYTHON=TRUE;;
        *) >&2 log_error  "Unsupported option: $1"
            print_help;;
	esac
	shift
done

for opt in VCPKG_PATH; do
    if ! [[ -v $opt ]]; then
        log_error "$opt is not set, please, read help!"
        print_help
    fi
done

cmake=${CMAKE_PATH:-$(which cmake)}
clang=${CLANG_PATH:-$(which clang)}
clangpp=${CLANGPP_PATH:-$(which clang++)}
vcpkg=$VCPKG_PATH
build_dir=${BUILD_DIR:-build}
build_type=${BUILD_TYPE:-Release}
enable_asan="${ENABLE_ASAN:-FALSE}"
build_tests="${BUILD_TESTS:-FALSE}"
build_examples="${BUILD_EXAMPLES:-FALSE}"
build_python="${BUILD_PYTHON:-FALSE}"

$cmake \
    --no-warn-unused-cli \
    -DCMAKE_BUILD_TYPE:STRING=$build_type \
    -DCMAKE_TOOLCHAIN_FILE:STRING=$vcpkg \
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
    -DCMAKE_C_COMPILER:FILEPATH=$clang \
    -DCMAKE_CXX_COMPILER:FILEPATH=$clangpp \
    -DNLD_BUILD_TESTS:BOOL=$build_tests \
    -DNLD_BUILD_EXAMPLES:BOOL=$build_examples \
    -DNLD_BUILD_PYTHON:BOOL=$build_python \
    -DENABLE_ASAN:BOOL=$enable_asan \
    -B $build_dir \
    -G "Unix Makefiles"

$cmake --build $build_dir --config $build_type --target all

