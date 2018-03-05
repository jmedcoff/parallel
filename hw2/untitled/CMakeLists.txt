cmake_minimum_required(VERSION 3.9)
project(untitled C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")

add_executable(untitled main.c)