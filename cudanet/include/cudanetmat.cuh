/*
 * ---------------------------------------------------------------------------
 * Copyright 2014 Nervana Systems Inc.  All rights reserved.
 * ---------------------------------------------------------------------------
 */
#define SYNC_THREADS 1

#define ERROR_INCOMPATIBLE_DIMENSIONS -1
#define CUBLAS_ERROR -2
#define CUDA_ERROR -3
#define VIEW_ERROR -4
#define ERROR_TRANSPOSED -5
#define ERROR_GENERIC -6
#define ERROR_TRANSPOSEDNESS -7
#define ERROR_NOT_ON_DEVICE -8
#define ERROR_UNSUPPORTED -9
#define ERROR_CONV_DIMENSION -10
#define ERROR_CONV_NUM_FILTERS -11
#define ERROR_AXIS_TYPE -12
#define ERROR_RND_NOT_INITIALIZED -13

struct cudanetmat {
    Matrix *data_host;
    NVMatrix *data_device;
    int on_device;
    int on_host;
    int size[2];
    int is_trans; // 0 or 1
    int owns_data;
};


struct rnd_struct {
    unsigned int* dev_mults;
    unsigned long long* dev_words;
};
