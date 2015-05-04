# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
cuda_add_library(cconv2_cudaconv SHARED
  src/conv_util.cu
  src/filter_acts.cu
  src/img_acts.cu
  src/weight_acts.cu
)

target_link_libraries(cconv2_cudaconv
  cconv2_utilpy
  cconv2_nvmatrix
  ${BLAS_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${PYTHON_LIBRARIES}
)

install(TARGETS cconv2_cudaconv
  LIBRARY DESTINATION lib
)
