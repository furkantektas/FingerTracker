cmake_minimum_required(VERSION 2.8.4)
project(FingerTracking)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    calibration.cpp
    calibration.h
    colors.h
    common_utils.cpp
    common_utils.h
    datatypes.h
    handDetection.cpp
    handDetection.h
    hand3D.cpp
    hand3D.h
    main.cpp)

find_package( OpenCV REQUIRED )
include_directories(${OPENCV_INCLUDE_DIR})

add_executable(FingerTracking ${SOURCE_FILES})

target_link_libraries(
  FingerTracking
  opencv_core
  opencv_highgui
  opencv_features2d
  opencv_calib3d
  opencv_video
  opencv_imgproc
)
