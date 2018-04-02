cmake_minimum_required (VERSION 2.8.11)
project (CS687)

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -D _USE_MULTITHREADING -O3 ")

file(GLOB_RECURSE SRCS *.cpp)

include_directories(../include)
include_directories(../header)

add_executable (pdt ${SRCS} )

