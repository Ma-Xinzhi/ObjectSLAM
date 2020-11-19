cmake_minimum_required(VERSION 2.8)
project(ObjectSLAM)

#SET(CMAKE_BUILD_TYPE Release)
#MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.0 REQUIRED)
if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 3.0 not found.")
endif()
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

# glog
find_package(Glog REQUIRED)
# gflags
find_package(GFlags REQUIRED)
# gtest
find_package(GTest REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${GTEST_INCLUDE_DIRS}
        ${GFLAGS_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_library(${PROJECT_NAME} SHARED
        src/utils/matrix_utils.cpp
        src/utils/dataprocess_utils.cpp
        src/Quadric.cpp
        src/MapDrawer.cpp
        src/Viewer.cpp
        src/FrameDrawer.cpp
        src/Map.cpp
        src/Frame.cpp
        src/Tracking.cpp
        src/BasicQuadricEdges.cpp
        src/InitializeQuadric.cpp
        src/System.cpp
        src/Dataset.cpp
        src/Detector.cpp
        src/ORBextractor.cpp
        src/MapPoint.cpp src/KeyFrame.cpp)

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${EIGEN3_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/Yolo/libdarknet.so
        ${Pangolin_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        ${GTEST_LIBRARIES}
        pthread
        )

target_link_libraries(${PROJECT_NAME} ${THIRD_PARTY_LIBS})

#FIND_PACKAGE( PCL REQUIRED )
#list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

#ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
#INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
#LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

# interface
#add_executable(rgbd
#        ./Example/interface/rgbd.cpp
#        )
#target_link_libraries(rgbd
#        tum_rgbd
#        boost_system
#        EllipsoidSLAM
#        ${PCL_LIBRARIES}
#        )

add_subdirectory(test)
add_subdirectory(Example)