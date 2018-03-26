cmake_minimum_required(VERSION 3.9)
project(neuralC)

set(CMAKE_CXX_STANDARD 11)

include_directories(.
                    ./pcg)

add_executable(
        potentials.h
        perceptron.h
        perceptron.cxx
        topology.h
        topology.cxx
)