PROJECT(Apps
    DESCRIPTION "Numerical applications and model problems using fast tensor product Schwarz smoothers on GPU."
)

message(STATUS "Configuring numerical applications within subdirectory 'apps'")

INCLUDE_DIRECTORIES(
        include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
)

SET( SOURCE_FILES
        stokes.cu
        benchmark_mg.cu
        )

FOREACH ( sourcefile ${SOURCE_FILES} )
    # string replace: cut off .cu from files
    STRING( REPLACE ".cu" "" testname ${sourcefile} )
    ADD_EXECUTABLE( ${testname} ${sourcefile})
    TARGET_COMPILE_OPTIONS(${testname} PRIVATE -fopenmp)
    TARGET_LINK_LIBRARIES(${testname} GPUTPS)
    TARGET_LINK_LIBRARIES(${testname} OpenMP::OpenMP_CXX)
    DEAL_II_SETUP_TARGET(${testname})
ENDFOREACH ( sourcefile ${APP_SOURCES} )