# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
cuda_add_library(cconv2_ConvNet SHARED
  src/actbroadcaster.cu
  src/data.cu
  src/layer_kernels.cu
  src/pyconvnet.cu
  src/weights.cu
  src/convnet.cu
  src/gradreducer.cu
  src/lr.cu
  src/reducepipeline.cu
  src/worker.cu
  src/copypipeline.cu
  src/jpeg.cpp
  src/memorysource.cu
  src/streambroadcast.cu
  src/cost.cu
  src/layer.cu
  src/neuron.cu
  src/util.cu
)

target_link_libraries(cconv2_ConvNet
  cconv2_utilpy
  cconv2_nvmatrix
  cconv2_cudaconv
  ${BLAS_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${JPEG_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

install(TARGETS cconv2_ConvNet
  LIBRARY DESTINATION lib
)
