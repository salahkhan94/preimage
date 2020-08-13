cmake_minimum_required(VERSION 2.8)
project(preimage)

#set(CMAKE_BUILD_TYPE "Release")
#set(G2O_USE_OPENGL false)
#add_definitions("-DENABLE_SSE")
#set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules/)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(CSparse REQUIRED)
Find_Package(Ceres REQUIRED)
find_package(OpenCV 3 REQUIRED)

LIST(APPEND G2O_LIBS
  cxsparse
  cholmod
  g2o_cli g2o_simulator
  g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d
  g2o_core g2o_solver_csparse g2o_solver_structure_only
  g2o_types_sba g2o_types_slam3d g2o_csparse_extension
  g2o_solver_dense g2o_stuff
  g2o_types_sclam2d g2o_parser g2o_solver_pcg
  g2o_types_data g2o_types_sim3
)
include_directories(
        ${PROJECT_SOURCE_DIR}
        ${EIGEN_INCLUDE_DIRS}
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

add_executable(ba src/g2o_bundle_adjustment.cpp)
target_link_libraries(ba ${G2O_LIBS} ${OpenCV_LIBS} utils)

add_executable(nonlin_triangulation src/nonlinear_triangulation.cpp)
target_link_libraries(nonlin_triangulation ${CERES_LIBRARIES} ${OpenCV_LIBS} utils)