# cuda-convnet2
Nervana's fork of Alex Krizhevsky's
[cuda-convnet2](https://code.google.com/p/cuda-convnet2/) containing several
extensions including:

* new python backend called cudanet for integration into Nervana's
  [neon](https://github.com/nervanaSystems/neon) framework
* several new kernels and functions to support things like multiway costs,
  python interface to GPU memory, support for non-texture kernels, array and
  scalar max/min comparisons, local contrast normalization.
* one line pip or cmake based installation
* additional checking and fixes.

## Installation ##

First ensure that you have met all required depdendency packges, as described 
on the cuda-convnet2
[compilation](https://code.google.com/p/cuda-convnet2/wiki/Compiling) page.

    # Clone this repository.
    git clone git@github.com:NervanaSystems/cuda-convnet2.git
    cd cuda-convnet2
    mkdir build
    cd build
    cmake ..
    make install  #for system-wide install, or else just make

The libraries will be added in the /usr/local/lib/ location. This path needs
to be in the `LD_LIBRARY_PATH` environment variable.

## Troubleshooting ##

If there are issues with finding `helper_cuda.h` add it to the paths:

    cmake -D CUDA_COMMON_INCLUDE_DIRS=[helper_cuda_path] -D CUDA_SDK_SEARCH_PATH=[helper_cuda_path] ..
    or directly in the top-level CMakeLists.txt in find_path(CUDA_COMMON_INCLUDE_DIRS 
     helper_cuda.h ... )
  
If there are issues with linking OpenCV:
Change the following:
in make-data/pyext/CMakeLists.txt 

    set(OpenCV_LIBRARIES "-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann -I/usr/include/opencv2 -L/usr/lib")

If there are issues opening `libcconv2_cudanet.so` make sure the permissions of 
the libraries in /usr/local/lib/ are set correctly. 
