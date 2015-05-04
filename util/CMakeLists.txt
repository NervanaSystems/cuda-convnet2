# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
add_library(cconv2_utilpy SHARED
  src/matrix.cpp
)

target_link_libraries(cconv2_utilpy
  ${BLAS_LIBRARIES}
  ${CBLAS_LIBRARIES}
  ${PYTHON_LIBRARIES}
)

install(TARGETS cconv2_utilpy
  LIBRARY DESTINATION lib
)
