PROJECT(Tests
    DESCRIPTION "Numerical applications and model problems using fast tensor product Schwarz smoothers on GPU."
)

message(STATUS "Configuring numerical tests within subdirectory 'tests'")

INCLUDE_DIRECTORIES(
        include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
)

SET( SOURCE_FILES
        operator_stokes.cu
        get_1d_matrices.cu
        patch_dofs.cu
        )

FOREACH ( sourcefile ${SOURCE_FILES} )
    # string replace: cut off .cu from files
    STRING( REPLACE ".cu" "" testname ${sourcefile} )
    ADD_EXECUTABLE( ${testname} ${sourcefile})
    DEAL_II_SETUP_TARGET(${testname})
    TARGET_LINK_LIBRARIES(${testname} GPUTPS)
ENDFOREACH ( sourcefile ${TEST_SOURCES} )