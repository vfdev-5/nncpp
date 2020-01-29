project(nncpp)
MESSAGE(STATUS "Project: ${PROJECT_NAME}")

cmake_minimum_required(VERSION 3.10)

#--- Create NNCpp library ---
find_package(CUDA REQUIRED)
MESSAGE(STATUS "Cuda found ${CUDA_FOUND}")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11 -O2")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

if(CUDA_VERSION GREATER 7.6)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
endif()

if(CUDA_VERSION GREATER 8.9)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
endif()

file(GLOB_RECURSE NNCPP_HEADERS nncpp/*.hpp)
file(GLOB_RECURSE NNCPP_CPP_SOURCES nncpp/*.cpp)
file(GLOB_RECURSE NNCPP_CU_HEADERS nncpp/*.cuh)
file(GLOB_RECURSE NNCPP_CU_SOURCES nncpp/*.cu)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nncpp)
CUDA_ADD_LIBRARY(${PROJECT_NAME} SHARED ${NNCPP_HEADERS} ${NNCPP_CU_HEADERS} ${NNCPP_CU_SOURCES} ${NNCPP_CPP_SOURCES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${CUDA_curand_LIBRARY})


#--- Setup tests ---
enable_testing()
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

file(GLOB_RECURSE TEST_SOURCES tests/*.cpp)

add_executable(test-${PROJECT_NAME} ${TEST_SOURCES})
include_directories(${GTEST_INCLUDE_DIRS})
target_link_libraries(test-${PROJECT_NAME} ${PROJECT_NAME} ${GTEST_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
