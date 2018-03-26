cmake_minimum_required(VERSION 2.8)
project(match)

find_package(OpenCV REQUIRED)

set(CMAKE_C_STANDARD 99)

add_executable( match test.cpp sim.cpp sim.hpp)
target_link_libraries( match ${OpenCV_LIBS})