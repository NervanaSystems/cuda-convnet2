# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
cmake_policy(VERSION 2.6)
if (POLICY CMP0054)
  cmake_policy(SET CMP0054 OLD)
endif (POLICY CMP0054)
project(cudaconvnet2)

set(CMAKE_BUILD_TYPE Release) # ensures -O3, no debugging symbols
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
endif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")

# update search path to include our cmake modules
set(CMAKE_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SCRIPT_DIR})

# find required dependencies
find_package(CUDA REQUIRED)
find_path(CUDA_COMMON_INCLUDE_DIRS
  helper_cuda.h
  PATHS ${CUDA_SDK_SEARCH_PATH}
    "/usr/local/cuda"
    "/Developer/NVIDIA/CUDA-6.5"
  PATH_SUFFIXES "samples/common/inc"
  DOC "Location of helper_cuda.h"
  NO_DEFAULT_PATH
)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}
             REQUIRED)

find_package(BLAS REQUIRED)
find_path(BLAS_INCLUDE_DIRS cblas.h /System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/)
find_package(CBLAS REQUIRED)

find_package(JPEG REQUIRED)

find_package(Threads REQUIRED)

# FindOpenCV.cmake exists in Linux, and homebrew OSX.  Not sure about anaconda
# OSX?
#find_package(OpenCV REQUIRED)

# no Numpy package in cmake, locate header dir manually
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
  "import numpy as n; print(n.get_include());"
  RESULT_VARIABLE _NUMPY_SEARCH_SUCCESS
  OUTPUT_VARIABLE _NUMPY_VALUES
  ERROR_VARIABLE _NUMPY_ERROR_VALUE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(_NUMPY_SEARCH_SUCCESS MATCHES 0)
  string(REGEX REPLACE ";" "\\\\;" _NUMPY_VALUES ${_NUMPY_VALUES})
  string(REGEX REPLACE "\n" ";" _NUMPY_VALUES ${_NUMPY_VALUES})
  list(GET _NUMPY_VALUES 0 NUMPY_INCLUDE_DIRS)
  string(REGEX REPLACE "\\\\" "/" NUMPY_INCLUDE_DIRS ${NUMPY_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "NumPy import failure:\n${_NUMPY_ERROR_VALUE}")
endif()


# setup compilation flags
list(APPEND CMAKE_CXX_FLAGS -DNUMPY_INTERFACE)
list(APPEND CUDA_NVCC_FLAGS -DNUMPY_INTERFACE)
if(APPLE)
  list(APPEND CUDA_NVCC_FLAGS -ccbin /usr/bin/clang -D_OSX=1)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -D_OSX=1")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++")
  set(CMAKE_MACOSX_RPATH True)
endif(APPLE)


# CUDA code generation flags
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)


include_directories(
  ${BLAS_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_COMMON_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIRS}/numpy
  util/include
  nvmatrix/include
  cudaconv3/include
  cudaconvnet/include
  cudanet/include
)


# compile targets for each subdirectory
add_subdirectory(util)
add_subdirectory(nvmatrix)
add_subdirectory(cudaconv3)
add_subdirectory(cudaconvnet)
add_subdirectory(cudanet)

CUDA_BUILD_CLEAN_TARGET()
