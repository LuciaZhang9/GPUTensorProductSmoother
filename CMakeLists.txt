message("                                                            ")
message("                GPU Tensor Product Smoothers                ")
message("                                                            ")

cmake_minimum_required(VERSION 2.12.0)

## Run "cmake --help-policy CMP0048" for policy details.
if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(GPUTPS
  LANGUAGES CXX
  VERSION 1.0
  DESCRIPTION "GPU Tensor Product Smoothers"
  )

################################################################################

if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  ## Extract the name of the current git branch.
  execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  ## Extract the abbreviated hash of the last commit.
  execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(GIT_BRANCH "")
  set(GIT_COMMIT_HASH "")
endif(EXISTS "${CMAKE_SOURCE_DIR}/.git")

message(STATUS "Current git branch:      ${GIT_BRANCH}")
message(STATUS "Current git commit hash: ${GIT_COMMIT_HASH}")

configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/git_version.h.in
  ${CMAKE_BINARY_DIR}/include/git_version.h
  )
  

################################################################################

## Fill (python) scripts with CMake information.
configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/ct_parameter.py.in
  ${CMAKE_BINARY_DIR}/scripts/ct_parameter.py
  )
  
  
################################################################################

find_package (PythonInterp 3 REQUIRED)

## Generates the default GPUTPS compile time parameters.
execute_process (
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_BINARY_DIR}/scripts/ct_parameter.py -O ${CMAKE_BINARY_DIR}/include/ct_parameter.h
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)


################################################################################

## Set a default build type if none was specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified.")
  ## Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()


FIND_PACKAGE(deal.II 9.3 QUIET
        HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
        )
IF(NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR "\n"
            "*** Could not locate deal.II. ***\n\n"
            "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
            "or set an environment variable \"DEAL_II_DIR\" that contains this path."
            )
ENDIF()

IF(NOT DEAL_II_WITH_MPI)
    MESSAGE(FATAL_ERROR
            "\n-- deal.II was built without support for MPI!\n"
            )
ENDIF()

IF(NOT DEAL_II_WITH_CUDA)
    MESSAGE(FATAL_ERROR
            "\n-- deal.II was built without support for CUDA!\n"
            )
ENDIF()

IF(DEAL_II_WITH_CUDA)
        enable_language(CUDA)
        INCLUDE_DIRECTORIES(/usr/local/cuda-11.2/samples/common/inc/)
        INCLUDE_DIRECTORIES(/opt/bwhpc/common/devel/cuda/11.2.2/samples/common/inc/)
        INCLUDE_DIRECTORIES(/export/home/cucui/SimServ06/cuda-samples/Common/)
ENDIF()

find_package (OpenMP REQUIRED)

# Set the source files to be compiled
SET( TARGET_SRC
        source/cuda_mg_transfer.cu
        source/cuda_vector.cu
        source/transfer_internal.cc)

# Set the include directory and the name of the project
INCLUDE_DIRECTORIES(
        include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

ADD_LIBRARY(GPUTPS STATIC ${TARGET_SRC})


# Define custom targets to easily switch the build type:
ADD_CUSTOM_TARGET(debug
        COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
        COMMENT "Switch CMAKE_BUILD_TYPE to Debug, now type 'make' to build"
        )

ADD_CUSTOM_TARGET(release
        COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
        COMMENT "Switch CMAKE_BUILD_TYPE to Release, now type 'make' to build"
        )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

DEAL_II_SETUP_TARGET(GPUTPS)

IF(EXISTS ${CMAKE_SOURCE_DIR}/apps/CMakeLists.txt)
    ADD_SUBDIRECTORY(apps)
ENDIF()

# Set up unit tests
IF(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
    ENABLE_TESTING()
    ADD_SUBDIRECTORY(tests)
ENDIF()

# Doxygen
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/Doxyfile.in)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/Doxyfile.in 
    ${CMAKE_CURRENT_BINARY_DIR}/scripts/Doxyfile @ONLY)
    add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/scripts/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
    endif(DOXYGEN_FOUND)
endif()


MESSAGE("====================================================")
