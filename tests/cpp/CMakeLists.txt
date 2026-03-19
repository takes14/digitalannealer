add_executable(da_engine_tests
    engine_tests.cu
)

set_target_properties(da_engine_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF
)

target_link_libraries(da_engine_tests PRIVATE
    da_kernels
    gtest_main
)

add_test(NAME da_engine_tests COMMAND da_engine_tests)
