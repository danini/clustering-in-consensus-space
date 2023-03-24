cmake_minimum_required(VERSION 3.8)

project(MultiConsensusClustering LANGUAGES CXX)

# indicate if OPENMP should be enabled
option(USE_OPENMP "Use OPENMP" ON)

# indicate if the sample project should be generated
option(CREATE_SAMPLE_PROJECT "Create the Project for the traditional multi-model tests." ON)
option(CREATE_PYTHON_BINDING "Create Pyhton bindings for the project." ON)

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

# ==============================================================================
# Includes
# ==============================================================================
SET(SOURCE_DIR "src/pyconsensusclustering")

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
# Structure: Multi-Consensus Library
# ==============================================================================
# Tell cmake that headers are in alse in source_dir
include_directories (${SOURCE_DIR}/include)

# Set header files for the library
file(GLOB_RECURSE HDRS_MULTICONSENSUS
	"${SOURCE_DIR}/include/*.h"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_MULTICONSENSUS 
	"${SOURCE_DIR}/include/*.cpp"
)

add_library(${PROJECT_NAME} STATIC 
	${HDRS_MULTICONSENSUS} 
	${SRCS_MULTICONSENSUS})

target_link_libraries(${PROJECT_NAME} 
	${OpenCV_LIBS}
	Eigen3::Eigen
	GraphCutRANSAC
	gflags
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# ==============================================================================
# Structure: Python binding
# ==============================================================================
if (CREATE_PYTHON_BINDING)
	# Generate python module
	add_subdirectory(lib/pybind11)
	
	pybind11_add_module(pymulticonsensus ${SOURCE_DIR}/src/bindings.cpp ${SOURCE_DIR}/src/pymulticonsensus_python.cpp  ${HDRS_MULTICONSENSUS} ${SRCS_MULTICONSENSUS} )
	
	target_link_libraries(pymulticonsensus PRIVATE  
		${OpenCV_LIBS} 
		Eigen3::Eigen 
		gflags
		GraphCutRANSAC)
endif (CREATE_PYTHON_BINDING)
	
# ==============================================================================
# Structure: Applications
# ==============================================================================
if (CREATE_SAMPLE_PROJECT)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	add_executable(SampleProject
		examples/cpp_example.cpp)
		
	target_link_libraries(SampleProject PUBLIC
		${PROJECT_NAME}
		${OpenCV_LIBS}
		Eigen3::Eigen
		GraphCutRANSAC
		gflags
	)

	add_dependencies(SampleProject ${PROJECT_NAME})
endif (CREATE_SAMPLE_PROJECT)
