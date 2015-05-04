# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
cuda_add_library(cconv2_cudanet SHARED
  src/cudanetmat.cu
)

target_link_libraries(cconv2_cudanet
  cconv2_utilpy
  cconv2_nvmatrix
  cconv2_cudaconv
  cconv2_ConvNet
  ${BLAS_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${JPEG_LIBRARIES}
)

install(TARGETS cconv2_cudanet
  LIBRARY DESTINATION lib
)
