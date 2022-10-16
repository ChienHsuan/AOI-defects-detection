cmake_minimum_required(VERSION 3.18.3)

# Create project
project(AOI-defects-detection)
add_executable(AOI-defects-detection main.cpp)

# c++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")

# Find package
set(OpenCV_DIR "C:/opencv_3.4.12/install")
find_package(OpenCV REQUIRED)

# Additional include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Additional library directories
link_directories(${OpenCV_LIB_DIR})

# Additional dependencies
target_link_libraries(AOI-defects-detection ${OpenCV_LIBS})
