if (CMAKE_HOST_WIN32)
    # need at least CMake 3.25 for IntelLLVM support of IntelSYCL package on Windows
    cmake_minimum_required(VERSION 3.25)
else()
    # CMake 3.23.5 is the minimum recommended for IntelLLVM on Linux
    cmake_minimum_required(VERSION 3.23.5)
endif()

project(lithium-battery-pipeline  LANGUAGES CXX)

SET(OpenCV_DIR "/home/intel/Downloads/opencv-4.8.0/build")
set(CMAKE_CXX_COMPILER "icpx")


if(WIN32)
SET(OpenCV_DIR "C:/Users/CRB/Downloads/opencv_4.5.5/build")
endif()

# Enable IPP_TL Component.
add_definitions(-DIPP_ENABLED_THREADING_LAYER)

find_package(OpenCV REQUIRED)

SET(IPP_SHARED True)
SET(IPP_ARCH intel64)
SET(IPP_TL_VARIANT TBB)
find_package(IPP REQUIRED)
# find_package(IntelSYCL REQUIRED)

if(WIN32)
set(CMAKE_CXX_COMPILER "icx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Qtbb ")
else()
set(CMAKE_CXX_COMPILER "icpx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -march=native")
# Replace the line above with  any one of below 2 lines to enable sycl-targets to ADL-P iGPU, which
# enables SYCL AOT (ahead of time) feature.
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=intel_gpu_adl_p ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=spir64_gen -Xs '-device 0x46a6' ")

# Uncomment below line to enable sycl-targets to A700M dGPU, which enables SYCL AOT (ahead of time)
# feature.
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=spir64_gen -Xs '-device 0x5690' ")
endif()

add_definitions(-DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

message(STATUS "cmake current source dir is ${CMAKE_CURRENT_SOURCE_DIR}")

if(WIN32)
	include_directories(${OpenCV_INCLUDE_DIRS} "C:/Program Files (x86)/Intel/oneAPI/ipp/latest/include/ipp")
else()
	include_directories(${OpenCV_INCLUDE_DIRS} "/opt/intel/oneapi/ipp/latest/include/ipp")
endif()

set(util_srcs
    src/util.cpp
)

set(ipp_srcs
    src/ipp_sobel.cpp
    src/ipp_magnitude.cpp
)

set(common_srcs
    src/common/slog.hpp
    src/common/stimer.hpp
)


add_executable(test_lithium_pipeline "src/test.cpp" ${ipp_srcs} ${util_srcs}  ${common_srcs} )

target_link_libraries(test_lithium_pipeline  ${OpenCV_LIBS} ${IPP_LIBRARIES})

