# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(COMPRESSED_SUFFIX ".tgz")
if(WIN32)
  set(OPENCV_LIB "opencv-win-x64-3.4.16")
  set(COMPRESSED_SUFFIX ".zip")
elseif(APPLE)
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
    set(OPENCV_LIB "opencv-osx-arm64-3.4.16")
  else()
    set(OPENCV_LIB "opencv-osx-x86_64-3.4.16")
  endif()
elseif(ANDROID)
  # Use different OpenCV libs according to toolchain
  # gcc: OpenCV 3.x, clang: OpenCV 4.x
  if(ANDROID_TOOLCHAIN MATCHES "clang")
    set(OPENCV_LIB "opencv-android-4.6.0")
    set(OPENCV_ANDROID_SHARED_LIB_NAME "libopencv_java4.so")
  elseif(ANDROID_TOOLCHAIN MATCHES "gcc")
    set(OPENCV_LIB "opencv-android-3.4.16")
    set(OPENCV_ANDROID_SHARED_LIB_NAME "libopencv_java3.so")
  else()
    message(FATAL_ERROR "Only support clang/gcc toolchain, but found ${ANDROID_TOOLCHAIN}.")
  endif()  
elseif(IOS)
  message(FATAL_ERROR "Not support cross compiling for IOS now!")
# Linux
else()
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(OPENCV_LIB "opencv-linux-aarch64-3.4.14")
  else()
    set(OPENCV_LIB "opencv-linux-x64-3.4.16")
  endif()
  if(ENABLE_OPENCV_CUDA)
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
      message(FATAL_ERROR "Cannot set ENABLE_OPENCV_CUDA=ON while in linux-aarch64 platform.")
    endif()
    set(OPENCV_LIB "opencv-linux-x64-gpu-3.4.16")
  endif()
endif()

set(OPENCV_INSTALL_DIR ${THIRD_PARTY_PATH}/install/)
if(ANDROID)
  set(OPENCV_URL_PREFIX "https://bj.bcebos.com/fastdeploy/third_libs")
else() # TODO: use fastdeploy/third_libs instead.
  set(OPENCV_URL_PREFIX "https://bj.bcebos.com/paddle2onnx/libs")
endif()
set(OPENCV_URL ${OPENCV_URL_PREFIX}/${OPENCV_LIB}${COMPRESSED_SUFFIX})

if(OPENCV_DIRECTORY)
  message(STATUS "Use the opencv lib specified by user. The OpenCV path: ${OPENCV_DIRECTORY}")
  STRING(REGEX REPLACE "\\\\" "/" OPENCV_DIRECTORY ${OPENCV_DIRECTORY})
  # For Android, the custom path to OpenCV with JNI should look like: 
  # -DOPENCV_DIRECTORY=your-path-to/OpenCV-android-sdk/sdk/native/jni
  if(ANDROID)
    if(WITH_OPENCV_STATIC)
      set(OpenCV_DIR ${OPENCV_DIRECTORY})
      find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
      include_directories(${OpenCV_INCLUDE_DIRS})
      list(APPEND DEPEND_LIBS ${OpenCV_LIBS})
    else()
      set(OpenCV_INCLUDE_DIRS ${OPENCV_DIRECTORY}/include)
      get_filename_component(OpenCV_NATIVE_DIR ${OPENCV_DIRECTORY} DIRECTORY)
      set(OpenCV_LIBS_DIR ${OpenCV_NATIVE_DIR}/libs)
      include_directories(${OpenCV_INCLUDE_DIRS})
      add_library(external_opencv_java STATIC IMPORTED GLOBAL)
      set_property(TARGET external_opencv_java PROPERTY IMPORTED_LOCATION 
        ${OpenCV_LIBS_DIR}/${ANDROID_ABI}/${OPENCV_ANDROID_SHARED_LIB_NAME})
      list(APPEND DEPEND_LIBS external_opencv_java)
    endif()
  # Win/Linux/Mac
  else()
    find_package(OpenCV REQUIRED PATHS ${OPENCV_DIRECTORY})
    include_directories(${OpenCV_INCLUDE_DIRS})
    list(APPEND DEPEND_LIBS ${OpenCV_LIBS})
  endif()
else()
  message(STATUS "Use the default OpenCV lib from: ${OPENCV_URL}")
  if(ANDROID)
    if(WITH_OPENCV_STATIC)
      # When FastDeploy uses the OpenCV static library, there is no need to install OpenCV to FastDeploy thirds_libs
      download_and_decompress(${OPENCV_URL} ${CMAKE_CURRENT_BINARY_DIR}/${OPENCV_LIB}${COMPRESSED_SUFFIX} ${THIRD_PARTY_PATH})
      set(OpenCV_DIR ${THIRD_PARTY_PATH}/${OPENCV_LIB}/sdk/native/jni)
      find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
      include_directories(${OpenCV_INCLUDE_DIRS})
      list(APPEND DEPEND_LIBS ${OpenCV_LIBS})
      # Still need OpenCV headers now. TODO(qiuyanjun): May remove OpenCV headers.
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${OpenCV_DIR} 
        ${THIRD_PARTY_PATH}/install/${OPENCV_LIB}/sdk/native/jni)
    else()
      # Installing OpenCV shared lib to FastDeploy third_libs/install dir.
      download_and_decompress(${OPENCV_URL} ${CMAKE_CURRENT_BINARY_DIR}/${OPENCV_LIB}${COMPRESSED_SUFFIX} ${THIRD_PARTY_PATH}/install)
      set(OpenCV_DIR ${THIRD_PARTY_PATH}/install/${OPENCV_LIB}/sdk/native/jni)
      get_filename_component(OpenCV_NATIVE_DIR ${OpenCV_DIR} DIRECTORY)
      set(OpenCV_INCLUDE_DIRS ${OpenCV_DIR}/include)
      set(OpenCV_LIBS_DIR ${OpenCV_NATIVE_DIR}/libs)
      include_directories(${OpenCV_INCLUDE_DIRS})
      add_library(external_opencv_java STATIC IMPORTED GLOBAL)
      set_property(TARGET external_opencv_java PROPERTY IMPORTED_LOCATION 
        ${OpenCV_LIBS_DIR}/${ANDROID_ABI}/${OPENCV_ANDROID_SHARED_LIB_NAME})
      list(APPEND DEPEND_LIBS external_opencv_java)
    endif()
  # Win/Linux/Mac
  else()
    download_and_decompress(${OPENCV_URL} ${CMAKE_CURRENT_BINARY_DIR}/${OPENCV_LIB}${COMPRESSED_SUFFIX} ${THIRD_PARTY_PATH}/install/)
    set(OpenCV_DIR ${THIRD_PARTY_PATH}/install/${OPENCV_LIB}/)
    if (WIN32)
      set(OpenCV_DIR ${OpenCV_DIR}/build/)
    endif()
    find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
    include_directories(${OpenCV_INCLUDE_DIRS})
    list(APPEND DEPEND_LIBS ${OpenCV_LIBS})
  endif()
endif()
