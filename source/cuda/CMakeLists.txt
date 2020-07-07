if (CMAKE_CUDA_COMPILER)
    add_library(ariadne-cuda OBJECT
        cuda_lib.cu
    )
endif()