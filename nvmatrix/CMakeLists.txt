# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
cuda_add_library(cconv2_nvmatrix SHARED
  src/memory.cu
  src/nvmatrix.cu
  src/nvmatrix_kernels.cu
)

target_link_libraries(cconv2_nvmatrix
  cconv2_utilpy
  ${CUDA_CUBLAS_LIBRARIES}
  ${PYTHON_LIBRARIES}
)

install(TARGETS cconv2_nvmatrix
  LIBRARY DESTINATION lib
)
