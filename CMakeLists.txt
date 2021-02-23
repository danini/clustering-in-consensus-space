cmake_minimum_required(VERSION 3.8)

project(ProgressiveXPrime LANGUAGES CXX)

# indicate if OPENMP should be enabled
option(USE_OPENMP "Use OPENMP" ON)

# indicate if the sample project should be generated
option(CREATE_TRADITIONAL_TEST_PROJECT "Create the Project for the traditional multi-model tests." ON)
option(CREATE_LINE_SEGMENT_PROJECT "Create the Project for the multi line segment tests." ON)
option(CREATE_POSE_ESTIMATION_PROJECT "Create the Project for multi-homography to pose tests." ON)

SET (CMAKE_BUILD_TYPE "RELEASE")

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# ==============================================================================
# Check C++17 support
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-fpermissive")
# ==============================================================================
# Find OpenCV
# ==============================================================================
find_package(OpenCV)
if (OPENCV_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS}")
  	add_definitions(-DUSE_OPENMP)
	set(TRGT_LNK_LBS_ADDITIONAL OpenMP::OpenMP_CXX)
endif (OPENCV_FOUND)

# ==============================================================================
# Find Eigen
# ==============================================================================
find_package(Eigen3 REQUIRED)
set(EIGEN_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

# ==============================================================================
# Find GFlags
# ==============================================================================
find_package(gflags REQUIRED)
if(DEFINED gflags_VERSION_STRING OR DEFINED GFLAGS_VERSION_STRING)
	# different cmake variable with lower or upper case exist
	set(GFLAGS_INCLUDE_DIR ${GFLAGS_INCLUDE_DIR})
	set(GFLAGS_LIBRARIES ${GFLAGS_LIBRARIES})
  message( "Found Gflags ${GFLAGS_VERSION_STRING}" )
  message( STATUS "> GFLAGS_INCLUDE_DIR:   ${GFLAGS_INCLUDE_DIR}" )
  message( STATUS "> GFLAGS_LIBRARIES:   ${GFLAGS_LIBRARIES}" )
endif()

find_package(Sophus)
message(STATUS "Search Dependency: Sophus")

if(Sophus_FOUND)
	if (TARGET Sophus::Sophus)
		get_target_property(Sophus_INCLUDE_DIR Sophus::Sophus INTERFACE_INCLUDE_DIRECTORIES)
	else()
		find_path(SOPHUS_INCLUDE_DIR "sophus/common.hpp")
	endif()
	include_directories(AFTER ${Sophus_INCLUDE_DIR})
else()
    message(WARNING "Using Sophus is set, but Sophus is NOT found. Sophus is used only in the pose estimation example project which will, thus, not be built.")
    set(CREATE_POSE_ESTIMATION_PROJECT 0)
endif()

# ==============================================================================
# Find OpenMP
# ==============================================================================
find_package(OpenMP)
if (USE_OPENMP)
	if(NOT OPENMP_FOUND)
		message(FATAL_ERROR "OPENMP not found.")
	endif()
	add_definitions(-DUSE_OPENMP)
	set(TRGT_LNK_LBS_ADDITIONAL OpenMP::OpenMP_CXX)
endif (USE_OPENMP)

find_package(Ceres REQUIRED)

# ==============================================================================
# Includes
# ==============================================================================
SET(SOURCE_DIR "src/pyprogxprime")

include_directories (
	${PROJECT_SOURCE_DIR}/graph-cut-ransac/src/pygcransac/include
)

# ==============================================================================
# Structure: Graph-Cut RANSAC Library
# ==============================================================================
# Set header files for the library
file(GLOB_RECURSE HDRS_GraphCutRANSAC
	"graph-cut-ransac/src/pygcransac/include/*.h"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_GraphCutRANSAC 
	"graph-cut-ransac/src/pygcransac/include/*.cpp"
)

add_library(GraphCutRANSAC STATIC
	${HDRS_GraphCutRANSAC}
	${SRCS_GraphCutRANSAC}
)

target_link_libraries(GraphCutRANSAC
	${OpenCV_LIBS}
	Eigen3::Eigen
	gflags)

# ==============================================================================
# Structure: Progressive-X Prime Library
# ==============================================================================

# Tell cmake that headers are in alse in source_dir
include_directories (${SOURCE_DIR}/include)

# Set header files for the library
file(GLOB_RECURSE HDRS_PROGXPRIME
	"${SOURCE_DIR}/include/*.h"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_PROGXPRIME 
	"${SOURCE_DIR}/include/*.cpp"
)

# ==============================================================================
# Structure: Python binding
# ==============================================================================
# Generate python module
#add_subdirectory(graph-cut-ransac/lib/pybind11)
#
#pybind11_add_module(pyprogxprime 
#	${SOURCE_DIR}/src/bindings.cpp 
#	${SOURCE_DIR}/src/pyprogxprime_python.cpp
#	${HDRS_PROGXPRIME} 
#	${SRCS_PROGXPRIME})
#
#target_link_libraries(pyprogxprime PRIVATE  
#	${OpenCV_LIBS} 
#	Eigen3::Eigen
#	GraphCutRANSAC)

add_library(${PROJECT_NAME} STATIC 
	${HDRS_PROGXPRIME} 
	${SRCS_PROGXPRIME})

target_link_libraries(${PROJECT_NAME} 
	${OpenCV_LIBS}
	Eigen3::Eigen
	GraphCutRANSAC
	gflags
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
	
# ==============================================================================
# Structure: Applications
# ==============================================================================
if (CREATE_TRADITIONAL_TEST_PROJECT)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	add_executable(TraditionalTests
		examples/cpp_example.cpp)
		
	target_link_libraries(TraditionalTests PUBLIC
		${PROJECT_NAME}
		${OpenCV_LIBS}
		Eigen3::Eigen
		GraphCutRANSAC
		gflags
	)

	add_dependencies(TraditionalTests ${PROJECT_NAME})
endif (CREATE_TRADITIONAL_TEST_PROJECT)

if (CREATE_LINE_SEGMENT_PROJECT)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	add_executable(LineSegmentTests
		examples/example_line_segments.cpp)
		
	target_link_libraries(LineSegmentTests PUBLIC
		${PROJECT_NAME}
		${OpenCV_LIBS}
		Eigen3::Eigen
		GraphCutRANSAC
		gflags
	)

	add_dependencies(LineSegmentTests ${PROJECT_NAME})
endif (CREATE_LINE_SEGMENT_PROJECT)

if (CREATE_POSE_ESTIMATION_PROJECT)
	find_package(OpenMVG CONFIG REQUIRED)

	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	add_executable(PoseGraphTests
		examples/example_pose.cpp
		examples/pose_averaging/openMVG.hpp
		examples/pose_averaging/openMVG.cpp)
		
	target_link_libraries(PoseGraphTests PUBLIC
		${PROJECT_NAME}
		${OpenCV_LIBS}
		Eigen3::Eigen
		GraphCutRANSAC
		gflags
		#Sophus
		OpenMVG::openMVG_sfm
	)

	add_dependencies(PoseGraphTests ${PROJECT_NAME})
endif (CREATE_POSE_ESTIMATION_PROJECT)
