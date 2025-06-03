cmake_minimum_required(VERSION 3.5)

project(YOLOObjectTrack VERSION 0.1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA 12 REQUIRED)

set(CMAKE_CUDA_STANDARD 12)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# !CUDA

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

set(PROJECT_SOURCES
    Obj_track_Gpu.cpp
    inference.h
    inference.cpp
    sort.h
    sort.cpp
    kalman_box_tracker.h
    kalman_box_tracker.cpp
    kuhn_munkres.h
    kuhn_munkres.cpp
    )

add_executable(YOLOObjectTrack ${PROJECT_SOURCES})
target_link_libraries(YOLOObjectTrack ${OpenCV_LIBS})
