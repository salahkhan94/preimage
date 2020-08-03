cmake_minimum_required(VERSION 2.8)
project(preimage)

#set(CMAKE_BUILD_TYPE "Release")
#set(G2O_USE_OPENGL false)
#add_definitions("-DENABLE_SSE")
#set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(CSparse REQUIRED)
find_package(OpenCV 3 REQUIRED)

SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse g2o_solver_cholmod g2o_types_sba)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${CSPARSE_INCLUDE_DIR}
        ${CHOLMOD_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        "/usr/include/eigen3/"
)
add_library(utils src/utils.cpp)
add_executable(triangulate src/triangulate.cpp)
target_link_libraries(triangulate ${G2O_LIBS} ${OpenCV_LIBS} utils)

