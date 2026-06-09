cmake_minimum_required(VERSION 3.16)

project(EXAONE_CPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(exaone
    main.cpp
    src/exaone.cpp
 "src/backend/basic_arithmetic.cpp")

target_include_directories(exaone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
