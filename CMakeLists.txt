cmake_minimum_required(VERSION 3.5)
project(AR_DEPTH)

set(CMAKE_CXX_STANDARD 11)
set(OpenCV_DIR C:/Users/Arturs/Libs/opencv_build/install)
set(Eigen3_DIR C:/Users/Arturs/Libs/eigen-3.3.7/build)
find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
SET(CMAKE_CXX_FLAGS  "-O2")

add_executable(AR_DEPTH main.cpp src/util.cpp src/ColmapReader.cpp src/ARDepth.cpp)
target_link_libraries (AR_DEPTH ${OpenCV_LIBS})
target_link_libraries (AR_DEPTH Eigen3::Eigen)
