cmake_minimum_required(VERSION 3.16)
project(industrial_vision_demo VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/.deps/onnxruntime/onnxruntime-osx-arm64-1.30.0"
    CACHE PATH "ONNX Runtime C/C++ SDK directory")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
          HINTS "${ONNXRUNTIME_ROOT}/include")
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
             HINTS "${ONNXRUNTIME_ROOT}/lib")
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime SDK not found; install it separately and set ONNXRUNTIME_ROOT")
endif()
add_library(onnxruntime_sdk SHARED IMPORTED)
set_target_properties(onnxruntime_sdk PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")

add_library(aoi_core src/aoi.cpp src/score_provider.cpp src/onnx_provider.cpp
                     src/validation.cpp)
target_include_directories(aoi_core PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(aoi_core PUBLIC ${OpenCV_LIBS} onnxruntime_sdk)
target_compile_options(aoi_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(aoi_demo src/main.cpp)
target_link_libraries(aoi_demo PRIVATE aoi_core)

include(CTest)
if(BUILD_TESTING)
  add_executable(aoi_tests tests/aoi_tests.cpp)
  target_link_libraries(aoi_tests PRIVATE aoi_core)
  add_test(NAME aoi_tests COMMAND aoi_tests)
endif()
