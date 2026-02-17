cmake_minimum_required (VERSION 3.0)
project (zknn)
set(NAME "zknn")

find_path(GMP_INCLUDE_DIR NAMES gmp.h)
find_library(GMP_LIBRARIES NAMES gmp libgmp)
find_library(GMPXX_LIBRARIES NAMES gmpxx libgmpxx)

find_path(CMAKE_FOLDER NAMES cmake/emp-tool-config.cmake)
include(${CMAKE_FOLDER}/cmake/emp-base.cmake)

find_package(emp-ot REQUIRED)
include_directories(${EMP-OT_INCLUDE_DIRS})

ADD_SUBDIRECTORY(libff)
include_directories(libff)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

ENABLE_TESTING()
ADD_SUBDIRECTORY(test)
