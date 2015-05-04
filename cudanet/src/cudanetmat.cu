/*
 * ---------------------------------------------------------------------------
 * Copyright 2014 Nervana Systems Inc.  All rights reserved.
 * ---------------------------------------------------------------------------
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <arrayobject.h>
#include <assert.h>
#include <helper_cuda.h>
#include <cublas.h>
#include <time.h>
#include <vector>
#include <execinfo.h>
#include <signal.h>

#include "../../util/include/matrix.h"
#include "../../util/include/queue.h"
#include "../../nvmatrix/include/nvmatrix.cuh"
#include "../../cudaconv3/include/cudaconv2.cuh"
#include "../include/cudanetmat.cuh"
#include "../../cudaconvnet/include/layer_kernels.cuh"

extern "C" {
int elementwise_check3(cudanetmat* mat1, cudanetmat* mat2, cudanetmat* target);
int elementwise_check2(cudanetmat* mat1, cudanetmat* target);

/* ------------------------------ CUBLAS init/shutdown ------------------------------ */

inline bool check_cublas_error() {
    cublasStatus status = cublasGetError();

    return status != CUBLAS_STATUS_SUCCESS;
}

inline bool checkCUDAError() {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
        printf("%s\n", cudaGetErrorString( err));
    return cudaSuccess != err;
}

extern const char* get_last_cuda_error() {
    cudaError_t err = cudaGetLastError();

    return cudaGetErrorString( err);
}

extern int cublas_init() {

    if (NVMatrix::getNumCublasHandles() == 0) {
        NVMatrix::initCublas();
    }
    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}

extern int cublas_shutdown() {
    if (NVMatrix::getNumCublasHandles() > 0) {
        NVMatrix::destroyCublas();
        cudaThreadExit();
    }
    return 0;
}

extern void init_random(unsigned long long seed) {
    if (!NVMatrix::isRndInitialized())
        NVMatrix::initRandom(seed);
}

extern void init_random_no_seed() {
    if (!NVMatrix::isRndInitialized())
        NVMatrix::initRandom();
}

extern void destroy_random() {
    if (NVMatrix::isRndInitialized())
        NVMatrix::destroyRandom();
}

extern int get_device_id() {
    // DEVICE_HOST is -1
    // DEVICE_NULL is -2
    return NVMatrix::getDeviceID();
}

extern void sync_stream() {
    NVMatrix::syncStream(NVMatrix::getDefaultStream());
}

extern void set_device_id(int d) {
    NVMatrix::setDeviceID(d);
}

extern int get_peer_access(int srcDevice, int tgtDevice) {
    return NVMatrix::canAccessPeer(srcDevice, tgtDevice);
}

extern int cuda_set_device(int deviceId) {
    cudaSetDevice(deviceId);
    
    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

extern int get_num_devices(int* err_code) {
    int numd;
    *err_code = cudaGetDeviceCount(&numd);
    return numd;
}

/* ------------------------------ Utility routines ------------------------------ */

int elementwise_check3(cudanetmat* mat1, cudanetmat* mat2, cudanetmat* target) {
    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    return 0;
}

int elementwise_check2(cudanetmat* mat, cudanetmat* target) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    return 0;
}

extern int get_leading_dimension(cudanetmat* mat) {
    return mat->is_trans ? mat->size[1] : mat->size[0];
}

extern int get_nonleading_dimension(cudanetmat* mat) {
    return mat->is_trans ? mat->size[0] : mat->size[1];
}

extern void set_transpose(cudanetmat* mat, int is_trans) {
    mat->is_trans = is_trans;
}

inline char get_transpose_char(cudanetmat* mat) {
    return mat->is_trans ? 't' : 'n';
}

extern void cuda_sync_threads() {
    cudaThreadSynchronize();
}

/* ------------------------------ Allocating/moving data ------------------------------ */

extern int allocate_device_memory(cudanetmat* mat) {
    mat->data_device = new NVMatrix(mat->size[0], mat->size[1], mat->is_trans ? true : false);
    mat->on_device = 1;
    return 0;
}

extern int copy_to_host(cudanetmat* mat) {
    if (mat->on_device) {
         mat->data_device->copyToHost(*(mat->data_host));
    } else
       return ERROR_NOT_ON_DEVICE;
 
    return 0;
}

extern int set_host_mat(cudanetmat* mat, float *data) {
    if (mat->data_host)
        delete mat->data_host;
    mat->data_host = new Matrix(data, (int64) mat->size[0], (int64) mat->size[1], mat->is_trans);

    return 0;
}

extern int get_data_device_id(cudanetmat* mat) {
    if (!mat->on_device) {
        return ERROR_NOT_ON_DEVICE;
    }

    return mat->data_device->getDataDeviceID();
}

extern int copy_to_device(cudanetmat* mat) {
    if (!mat->on_device) {
        allocate_device_memory(mat);
        mat->data_device->copyFromHost(*(mat->data_host), true);
    } else {
        mat->data_device->copyFromHost(*(mat->data_host), true);
    }
    mat->is_trans = mat->data_device->isTrans();
    mat->size[0] = mat->data_device->getNumRows();
    mat->size[1] = mat->data_device->getNumCols();
    return 0;
}

extern int copy_from(cudanetmat* mat, float* data, bool is_trans) {
    Matrix mat_data(data, (int64) mat->size[0], (int64) mat->size[1], is_trans);
    mat->data_device->copyFromHost(mat_data, false);
    return 0;
}

extern int copy_on_device(cudanetmat* mat1, cudanetmat* mat2) {
    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat1->data_device->copy(*(mat2->data_device));
    return 0;
}


extern void init_from_array(cudanetmat* mat, float* data, int m, int n) {
    mat->data_host = new Matrix(data, (int64) m, (int64) n, false);
    mat->size[0] = m;
    mat->size[1] = n;
    mat->on_device = 0;
    mat->on_host = 1;
    mat->is_trans = 0;
    mat->owns_data = 1;
}

extern int init_empty(cudanetmat* mat, int m, int n) {
    mat->size[0] = m;
    mat->size[1] = n;
    mat->on_device = 0;
    mat->on_host = 0;
    mat->is_trans = 0;
    mat->owns_data = 1;

    return allocate_device_memory(mat);
}

extern int assign_scalar(cudanetmat* mat, float alpha) {

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;
    mat->data_device->assign(alpha);
    return 0;
}

extern int add_scalar(cudanetmat* mat, float alpha, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;
    mat->data_device->addScalar(alpha, *(target->data_device));
    return 0;
}

extern int add_elementwise(cudanetmat* mat1, cudanetmat* mat2, cudanetmat* target) {
    int errcheck = elementwise_check3(mat1, mat2, target);
    if (errcheck !=0) return errcheck;
    mat1->data_device->add(*(mat2->data_device), *(target->data_device));
    return 0;
}

extern int subtract_elementwise(cudanetmat* mat1, cudanetmat* mat2, cudanetmat* target) {
    int errcheck = elementwise_check3(mat1, mat2, target);
    if (errcheck !=0) return errcheck;
    mat1->data_device->subtract(*(mat2->data_device), *(target->data_device));
    return 0;
}

extern int divide_elementwise(cudanetmat* mat1, cudanetmat* mat2, cudanetmat* target) {
    int errcheck = elementwise_check3(mat1, mat2, target);
    if (errcheck !=0) return errcheck;
    mat1->data_device->eltwiseDivide(*(mat2->data_device), *(target->data_device));
    return 0;
}

/* Elementwise multiplication of 2 matrices */
extern int mult_elementwise(cudanetmat* mat1, cudanetmat* mat2, cudanetmat* target) {
    int errcheck = elementwise_check3(mat1, mat2, target);
    if (errcheck !=0) return errcheck;
    mat1->data_device->eltwiseMult(*(mat2->data_device), *(target->data_device));
    return 0;
}

extern int mult_by_scalar(cudanetmat* mat, float alpha, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;
    mat->data_device->scale(alpha, *(target->data_device));
    return 0;
}

extern int divide_by_scalar(cudanetmat* mat, float alpha, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;
    mat->data_device->apply(NVMatrixOps::DivByScalar(alpha), *(target->data_device));
    return 0;
}

extern int sign(cudanetmat* mat, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;
    mat->data_device->apply(NVMatrixOps::Sign(), *(target->data_device));
    return 0;
}
extern int apply_sigmoid(cudanetmat* mat, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;
    mat->data_device->apply(NVMatrixOps::Logistic(), *(target->data_device));
    return 0;
}
extern int apply_tanh(cudanetmat* mat, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;
    mat->data_device->apply(NVMatrixOps::Tanh(), *(target->data_device));
    return 0;
}
extern int apply_soft_threshold(cudanetmat* mat, float alpha, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;
    mat->data_device->apply(NVMatrixOps::SoftThreshold(alpha), *(target->data_device));
    return 0;
}
extern int apply_abs(cudanetmat* mat, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;
    mat->data_device->apply(NVMatrixOps::Abs(), *(target->data_device));
    return 0;
}
extern int apply_log_1_plus_exp(cudanetmat* mat, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;
    mat->data_device->apply(NVMatrixOps::Log1PlusExp(), *(target->data_device));
    return 0;
}
extern int apply_log(cudanetmat* mat, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;
    mat->data_device->apply(NVMatrixOps::Log(), *(target->data_device));
    return 0;
}

extern int apply_clip_range(cudanetmat* mat, cudanetmat* target, float lower, float upper) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans || target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    mat->data_device->apply(NVMatrixOps::ClipUpperLower(lower, upper), *(target->data_device));
    return 0;
}

extern int apply_exp(cudanetmat* mat, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;
    mat->data_device->apply(NVMatrixOps::Exp(), *(target->data_device));
    return 0;
}
extern int apply_gamma(cudanetmat* mat, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    mat->data_device->apply(NVMatrixOps::Gamma(), *(target->data_device));
    return 0;
}
extern int apply_lgamma(cudanetmat* mat, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    mat->data_device->apply(NVMatrixOps::LGamma(), *(target->data_device));
    return 0;
}
extern int apply_sqrt(cudanetmat* mat, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;
    mat->data_device->apply(NVMatrixOps::Sqrt(), *(target->data_device));
    return 0;
}
extern int apply_pow(cudanetmat* mat, float pow, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;
    mat->data_device->apply(NVMatrixOps::Pow(pow),*(target->data_device));
    return 0;
}

// For convolution, krizhevsky expects
// Weights as (K1xK2xC) Rows x F Columns in 'C' order
// Images as  (D1xD2xC) Rows x (N) Columns in 'C' order 
// Target as  (OD1xOD2xF) Rows x (N) Columsn in 'C' order

extern int convolution(cudanetmat* wts, cudanetmat* imgs, cudanetmat* targets, int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups, bool localconv) {
//    int numFilterColors = numImgColors / numGroups;      
    int numFilters = wts->size[1];
    int numModules = numModulesX * numModulesY;
    int numImages = imgs->size[1];
    int imgPixels = imgs->size[0]/numImgColors;
    int imgSizeX = imgPixels / imgSizeY;
    if (wts->is_trans || imgs->is_trans || targets->is_trans) {
        return ERROR_TRANSPOSEDNESS;
    }
    if (imgPixels != imgSizeY*imgSizeX) 
        return ERROR_CONV_DIMENSION;
    if (numFilters % 16 != 0) 
        return ERROR_CONV_NUM_FILTERS;
    if (targets->size[0] != numFilters * numModules || targets->size[1] != numImages) 
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (!localconv) {
        convFilterActs(*(imgs->data_device), *(wts->data_device), *(targets->data_device), imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numGroups);
    } else {
        localFilterActs(*(imgs->data_device), *(wts->data_device), *(targets->data_device), imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numGroups);        
    }
    return 0;
}

extern int convolution_back_weights(cudanetmat* hidActs, cudanetmat* imgs, cudanetmat* targets, int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups, int sumWidth, float scaleTargets, float scaleOutputs, bool localconv) {
//    int numFilterColors = numImgColors / numGroups;      
    int numFilters = targets->size[1];
    // int numModules = numModulesX * numModulesX;
    // int numImages = imgs->size[1];
    int imgPixels = imgs->size[0]/numImgColors;
    int imgSizeX = imgPixels / imgSizeY;
    int filterPixels = filterSize*filterSize;
    int filterChannels = numImgColors/numGroups;
    int outWidth = DIVUP(numModulesX, sumWidth);
    int outChunks = outWidth * outWidth;
    if (hidActs->is_trans || imgs->is_trans || targets->is_trans) {
        return ERROR_TRANSPOSEDNESS;
    }
    if (imgPixels != imgSizeY*imgSizeX) 
        return ERROR_CONV_DIMENSION;
    if (numFilters % 16 != 0) 
        return ERROR_CONV_NUM_FILTERS;

    if (!localconv) {
        if (targets->size[0] != filterChannels * filterPixels || targets->size[1] != numFilters) 
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        bool doPartialSum = sumWidth < numModulesX;
        NVMatrix _weightGradTmp;
        NVMatrix& tgt = doPartialSum ? _weightGradTmp : *(targets->data_device);
        convWeightActs(*(imgs->data_device), *(hidActs->data_device), tgt, imgSizeY, numModulesY, numModulesX, filterSize, paddingStart, 
            moduleStride, numImgColors, numGroups, sumWidth, doPartialSum ? 0 : scaleTargets, scaleOutputs);
        if (doPartialSum) {
            int pScaleTargets = scaleTargets > 0; // TODO determine whether this makes sense
            _weightGradTmp.reshape(outChunks, filterChannels * filterPixels * numFilters);
            targets->data_device->addSum(_weightGradTmp, 0, pScaleTargets, 1);
            targets->data_device->reshape(filterChannels * filterPixels, numFilters);
        }
    } else {
        localWeightActs(*(imgs->data_device), *(hidActs->data_device), *(targets->data_device), imgSizeY, numModulesY, numModulesX, filterSize, paddingStart, 
            moduleStride, numImgColors, numGroups, scaleTargets, scaleOutputs);                
    }


    return 0;
}


extern int convolution_back_errors(cudanetmat* wts, cudanetmat* imgs, cudanetmat* targets, int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups, float scaleTargets, bool localconv) {
    int numFilterColors = numImgColors / numGroups;      
    int numFilters = wts->size[1];
    // int numModules = numModulesX * numModulesX;
    int numImages = imgs->size[1];
    int numModules = imgs->size[0]/numFilters;
    if (wts->is_trans || imgs->is_trans || targets->is_trans) {
        return ERROR_TRANSPOSEDNESS;
    }
    int filterModuleMult = localconv ? 1 : numModules;
    int filterPixels = wts->size[0] / (filterModuleMult * numFilterColors);
    int filterSize = sqrt(filterPixels);
    int imgPixels = imgSizeY * imgSizeX;

    if (numFilters % 16 != 0) 
        return ERROR_CONV_NUM_FILTERS;
    if (targets->size[0] != numImgColors * imgPixels || targets->size[1] != numImages) 
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (!localconv) {
        convImgActs(*(imgs->data_device), *(wts->data_device), *(targets->data_device), imgSizeY, imgSizeX, numModulesY,
                    paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, 1);
    } else {
        localImgActs(*(imgs->data_device), *(wts->data_device), *(targets->data_device), imgSizeY, imgSizeX, numModulesY,
                    paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, 1);        
    }
    return 0;
}

extern int dot(cudanetmat* mat1, cudanetmat* mat2, cudanetmat* target, float beta, float alpha) {
    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (get_leading_dimension(mat1) != get_leading_dimension(target) ||
        get_nonleading_dimension(mat2) != get_nonleading_dimension(target) ||
        get_nonleading_dimension(mat1) != get_leading_dimension(mat2)) {
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    }
    int m = get_leading_dimension(mat1),
        k = get_leading_dimension(mat2),
        n = get_nonleading_dimension(mat2);

    // cublas, why?
    // had to do some weirdness here to avoid forcing target to transpose  (added function to nvmatrix to handle row major matrices)
    target->data_device->addProductRM(*(mat1->data_device), *(mat2->data_device), beta, alpha, mat1->is_trans, mat2->is_trans);
    return 0;
}

extern float vdot(cudanetmat* mat1, cudanetmat* mat2, int* err_code) {
    float res;

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans) {
        *err_code = ERROR_TRANSPOSEDNESS;
        return 0;
    }

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1]) { 
        *err_code = ERROR_INCOMPATIBLE_DIMENSIONS;
        return 0;
    }

    res = mat1->data_device->dotProduct(*(mat2->data_device));

    *err_code = 0;
    return res;
}

extern int add_vector(cudanetmat* mat, cudanetmat* vec, float scaleVec, cudanetmat* target) {
    if (!mat->on_device || !vec->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;

    if (target == vec) 
        return ERROR_UNSUPPORTED;

    if (vec->size[0] != 1 && vec->size[1] != 1)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (vec->size[0] != mat->size[0] && vec->size[1] != mat->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->data_device->addVector(*(vec->data_device), scaleVec, *(target->data_device));
    return 0;
}


extern int mat_vector_op(cudanetmat* mat, cudanetmat* vec, float scaleVec, cudanetmat* target, char opchar) {
    if (!mat->on_device || !vec->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;

    if (target == vec) 
        return ERROR_UNSUPPORTED;

    if (vec->size[0] != 1 && vec->size[1] != 1)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (vec->size[0] != mat->size[0] && vec->size[1] != mat->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    switch (opchar) {
        case 'a' :
            mat->data_device->addVector(*(vec->data_device), scaleVec, *(target->data_device));
            break;
        case 's' : 
            mat->data_device->addVector(*(vec->data_device), -scaleVec, *(target->data_device));
            break;
        case 'm' :
            mat->data_device->eltwiseMultByVector(*(vec->data_device), *(target->data_device));
            break;
        case 'd' :
            mat->data_device->eltwiseDivideByVector(*(vec->data_device), *(target->data_device));
            break;
        case 'e' :
            mat->data_device->equalsVector(*(vec->data_device), *(target->data_device));
            break;
        default: {
            printf("This char is unsupported: %c\n", opchar);
            return ERROR_UNSUPPORTED;
        }
    }
    return 0;
}

extern int quantize(cudanetmat* mat, int intwidth, int totalwidth) {
    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;
    mat->data_device->quantizeValues(intwidth, abs(intwidth-totalwidth));
    return 0;
}

extern int randomize_gaussian(cudanetmat* mat, float mean, float stdev) {
    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (!NVMatrix::isRndInitialized())
        return ERROR_RND_NOT_INITIALIZED;
    mat->data_device->randomizeGaussian(mean, stdev);
    return 0;
}

extern int randomize_uniform(cudanetmat* mat) {
    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (!NVMatrix::isRndInitialized())
        return ERROR_RND_NOT_INITIALIZED;
    mat->data_device->randomizeUniform();
    return 0;
}

extern int randomize_uniform_thresh(cudanetmat* mat, float thresh) {
    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (!NVMatrix::isRndInitialized())
        return ERROR_RND_NOT_INITIALIZED;
    mat->data_device->randomizeUniform();
    mat->data_device->apply(NVMatrixOps::DropoutKernelOperator(thresh));
    return 0;
}

extern int randomize_binary(cudanetmat* mat) {
    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (!NVMatrix::isRndInitialized())
        return ERROR_RND_NOT_INITIALIZED;
    mat->data_device->binarizeProbs();
    return 0;
}

extern int add_noise_gaussian(cudanetmat* mat, float stdev) {
    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (!NVMatrix::isRndInitialized())
        return ERROR_RND_NOT_INITIALIZED;
    mat->data_device->addGaussianNoise(stdev);
    return 0;
}

extern int add_noise_uniform(cudanetmat* mat, float minRange, float maxRange) {
    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (!NVMatrix::isRndInitialized())
        return ERROR_RND_NOT_INITIALIZED;
    mat->data_device->addUniformNoise(minRange, maxRange);
    return 0;
}

extern int unpool_forward(cudanetmat* smallMat, cudanetmat* largeMat, int channels, int sizeX, int smallX, int largeX) {
    if (!smallMat->on_device || !largeMat->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (smallMat->is_trans || largeMat->is_trans)
        return ERROR_TRANSPOSEDNESS;
    convLocalUnpoolForward(*(smallMat->data_device), *(largeMat->data_device), channels, sizeX, smallX, largeX);

    largeMat->size[0] = largeMat->data_device->getNumRows();
    largeMat->size[1] = largeMat->data_device->getNumCols();
    return 0;
}

extern int unpool_backward(cudanetmat* largeMat, cudanetmat* smallMat, int channels, int sizeX, int smallX, int largeX) {
    if (!smallMat->on_device || !largeMat->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (smallMat->is_trans || largeMat->is_trans)
        return ERROR_TRANSPOSEDNESS;
    convLocalUnpoolBackward(*(largeMat->data_device), *(smallMat->data_device), channels, sizeX, smallX, largeX);

    smallMat->size[0] = smallMat->data_device->getNumRows();
    smallMat->size[1] = smallMat->data_device->getNumCols();
    return 0;
}

extern int max_pool(cudanetmat* mat, cudanetmat* target, int channels, int sizeX, int start, int stride, int outputsX) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans || target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    convLocalPool(*(mat->data_device), *(target->data_device), channels, sizeX, start, stride, outputsX, MaxPooler());
    target->size[0] = target->data_device->getNumRows();
    target->size[1] = target->data_device->getNumCols();
    return 0;
}

extern int max_abs_pool(cudanetmat* mat, cudanetmat* target, int channels, int sizeX, int start, int stride, int outputsX) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans || target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    convLocalPool(*(mat->data_device), *(target->data_device), channels, sizeX, start, stride, outputsX, MaxAbsPooler());
    target->size[0] = target->data_device->getNumRows();
    target->size[1] = target->data_device->getNumCols();
    return 0;
}

extern int avg_pool(cudanetmat* mat, cudanetmat* target, int channels, int sizeX, int start, int stride, int outputsX) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans || target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    convLocalPool(*(mat->data_device), *(target->data_device), channels, sizeX, start, stride, outputsX, AvgPooler());
    target->size[0] = target->data_device->getNumRows();
    target->size[1] = target->data_device->getNumCols();
    return 0;
}

extern int l2_pool(cudanetmat* mat, cudanetmat* target, int channels, int sizeX, int start, int stride, int outputsX) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans || target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    convLocalPool(*(mat->data_device), *(target->data_device), channels, sizeX, start, stride, outputsX, L2Pooler());
    target->size[0] = target->data_device->getNumRows();
    target->size[1] = target->data_device->getNumCols();
    return 0;
}

extern int max_pool_undo(cudanetmat* imgs, cudanetmat* maxGrads, cudanetmat* maxActs, cudanetmat* target, int sizeX, int start, int stride, int outputsX) {
    if (!imgs->on_device || !maxGrads->on_device || !maxActs->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (imgs->is_trans || maxGrads->is_trans || maxActs->is_trans || target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (maxGrads->size[0]!=maxActs->size[0] || maxGrads->size[1] != maxActs->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (imgs->size[0]!=target->size[0] || imgs->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    convLocalMaxUndo(*(imgs->data_device), *(maxGrads->data_device), *(maxActs->data_device), *(target->data_device), sizeX, start, stride, outputsX);
    target->size[0] = target->data_device->getNumRows();
    target->size[1] = target->data_device->getNumCols();
    return 0;
}

extern int avg_pool_undo(cudanetmat* avgGrads, cudanetmat* target, int sizeX, int start, int stride, int outputsX, int imgSizeX) {
    if (!avgGrads->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (avgGrads->is_trans || target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    convLocalAvgUndo(*(avgGrads->data_device), *(target->data_device), sizeX, start, stride, outputsX, imgSizeX);
    target->size[0] = target->data_device->getNumRows();
    target->size[1] = target->data_device->getNumCols();
    return 0;
}

extern int l2_pool_undo(cudanetmat* imgs, cudanetmat* l2Grads, cudanetmat* l2Acts, cudanetmat* target, int sizeX, int start, int stride, int outputsX) {
    if (!imgs->on_device || !l2Grads->on_device || !l2Acts->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (imgs->is_trans || l2Grads->is_trans || l2Acts->is_trans || target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (l2Grads->size[0]!=l2Acts->size[0] || l2Grads->size[1] != l2Acts->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (imgs->size[0]!=target->size[0] || imgs->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    convLocalL2Undo(*(imgs->data_device), *(l2Grads->data_device), *(l2Acts->data_device), *(target->data_device), sizeX, start, stride, outputsX);
    target->size[0] = target->data_device->getNumRows();
    target->size[1] = target->data_device->getNumCols();
    return 0;
}

extern int crossmap_response_norm(cudanetmat* mat, cudanetmat* target, int channels, int sizeX, float scale, float power) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans || target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    convResponseNormCrossMap(*(mat->data_device),  *(target->data_device), channels, sizeX, scale, power, 1.0, false);
    target->size[0] = target->data_device->getNumRows();
    target->size[1] = target->data_device->getNumCols();
    return 0;
}
// v = respGrads, inputs = imgs, getActs = respActs
// convResponseNormUndo(v, _denoms, *_inputs[0], getActs(), _prev[replicaIdx][0]->getActsGrad(), _channels, _size, _scale, _pow, scaleTargets, 1);
extern int crossmap_response_norm_undo(cudanetmat* imgs, cudanetmat* respGrads, cudanetmat* respActs, cudanetmat* target, int channels, int sizeX, float scale, float power, float scaleTargets) {
    if (!imgs->on_device || !respGrads->on_device || !respActs->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (imgs->is_trans || respGrads->is_trans || respActs->is_trans || target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (respGrads->size[0]!=respActs->size[0] || respGrads->size[1] != respActs->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (imgs->size[0]!=target->size[0] || imgs->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    convResponseNormCrossMapUndo(*(respGrads->data_device), *(imgs->data_device), *(respActs->data_device), *(target->data_device), 
                                 channels, sizeX, scale, power, 1.0, false, scaleTargets, 1);
    target->size[0] = target->data_device->getNumRows();
    target->size[1] = target->data_device->getNumCols();
    return 0;
}

extern int local_contrast_norm(cudanetmat* mat, cudanetmat* meanDiffs, cudanetmat *denoms, cudanetmat* target, int imgSizeX, int channels, int sizeX, float scale, float power) {
    if (!meanDiffs->on_device || !denoms->on_device || !mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (mat->is_trans || target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    convLocalPool(*(mat->data_device), *(meanDiffs->data_device), channels, sizeX, -sizeX/2, 1, imgSizeX, AvgPooler());
    meanDiffs->data_device->add(*(mat->data_device), -1, 1);
    convContrastNorm(*(mat->data_device), *(meanDiffs->data_device), *(denoms->data_device), *(target->data_device), channels, sizeX, scale, power, 1.0);

    target->size[0] = target->data_device->getNumRows();
    target->size[1] = target->data_device->getNumCols();
    return 0;
}

// convContrastNormUndo(v, _denoms, _meanDiffs, getActs(), _prev[replicaIdx][inpIdx]->getActsGrad(), _channels, _size, _scale, _pow, scaleTargets, 1);
extern int local_contrast_norm_undo(cudanetmat* meanDiffs, cudanetmat *denoms, cudanetmat* respGrads, cudanetmat* respActs, cudanetmat* target, int channels, int sizeX, float scale, float power, float scaleTargets) {
    if (!meanDiffs->on_device || !denoms->on_device || !respGrads->on_device || !respActs->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;
    if (respGrads->is_trans || respActs->is_trans || target->is_trans)
        return ERROR_TRANSPOSEDNESS;
    if (respGrads->size[0]!=respActs->size[0] || respGrads->size[1] != respActs->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    if (meanDiffs->size[0]!=target->size[0] || meanDiffs->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    convContrastNormUndo(*(respGrads->data_device), *(denoms->data_device), *(meanDiffs->data_device), *(respActs->data_device), *(target->data_device),
                    channels, sizeX, scale, power, scaleTargets, 1);

    target->size[0] = target->data_device->getNumRows();
    target->size[1] = target->data_device->getNumCols();
    return 0;
}

extern int adadelta_update(cudanetmat* grads, cudanetmat* eGradSq, cudanetmat* eDeltSq, cudanetmat* deltX, float rho, float eps){
    int errcheck = elementwise_check3(grads, eGradSq, eDeltSq);
    if (errcheck !=0) return errcheck;

    errcheck = elementwise_check2(grads, deltX);
    if (errcheck !=0) return errcheck;

    // This operator is used to compute the decay updates:  a(t) = a(t-1) * rho + b(t)*b(t) * (1-rho)
    NVMatrixBinaryOps::AxPBysq sqwadd = NVMatrixBinaryOps::AxPBysq(rho, 1-rho);
    NVMatrixTernaryOps::SqrtRatioMult srmult = NVMatrixTernaryOps::SqrtRatioMult(eps);

    eGradSq->data_device->applyBinary(sqwadd, *(grads->data_device));
    eDeltSq->data_device->applyTernary(srmult, *(eGradSq->data_device), *(grads->data_device), *(deltX->data_device));
    eDeltSq->data_device->applyBinary(sqwadd, *(deltX->data_device));
    return 0;
}

extern int get_vector_slice(cudanetmat* source, cudanetmat* target, unsigned int first_ind, unsigned int last_ind) {
    // source must be a vector
    if (source->size[0] > 1 && source->size[1] > 1)
        return ERROR_GENERIC;

    if (source->is_trans)
        return ERROR_TRANSPOSED;

    if (!source->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (first_ind >= last_ind)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (source->size[0] > 1) {
        //source is a column vect
        if (last_ind > source->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        target->size[0] = last_ind - first_ind;
        target->size[1] = 1;
        target->data_device = &(source->data_device->slice(first_ind, last_ind, 0,1));
    } else {
        if (last_ind > source->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        //source is a row vect
        target->size[0] = 1;
        target->size[1] = last_ind - first_ind;
        target->data_device = &(source->data_device->slice(0,1,first_ind, last_ind));
    }
    target->on_device = 1;
    target->on_host = 0;
    target->is_trans = 0;
    target->owns_data = 0;
    return 0;
}

extern int get_slice_view(cudanetmat* source, cudanetmat* target, unsigned int first_row, unsigned int last_row, unsigned int first_col, unsigned int last_col) {
    if (source->is_trans)
        return ERROR_TRANSPOSED;

    if (!source->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (last_col > source->size[1] || (first_col >= last_col))
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (last_row > source->size[0] || (first_row >= last_row))
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    target->data_device = &(source->data_device->slice(first_row, last_row, first_col, last_col));
    target->data_host = NULL;
    target->on_device = 1;
    target->on_host = 0;
    target->size[0] = last_row - first_row;
    target->size[1] = last_col - first_col;
    target->is_trans = 0;
    target->owns_data = 0;

    return 0;
}

extern int get_col_slice_view(cudanetmat* source, cudanetmat* target, unsigned int first_col, unsigned int last_col) {
    return get_slice_view(source, target, 0, source->size[0], first_col, last_col);
}

extern int get_row_slice_view(cudanetmat* source, cudanetmat* target, unsigned int first_row, unsigned int last_row) {
    return get_slice_view(source, target, first_row, last_row, 0, source->size[1]);
}

extern int get_col_slice_copy(cudanetmat* source, cudanetmat* target, unsigned int first_col, unsigned int last_col) {
    if (source->is_trans)
        return ERROR_TRANSPOSED;

    if (!source->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (last_col > source->size[1] || (first_col >= last_col))
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    source->data_device->sliceCols(first_col, last_col, *(target->data_device));
    target->on_device = 1;
    target->on_host = 0;
    target->size[0] = source->size[0];
    target->size[1] = last_col - first_col;
    target->is_trans = 0;
    target->owns_data = 1;

    return 0;
}

extern int get_row_slice_copy(cudanetmat* source, cudanetmat* target, unsigned int first_row, unsigned int last_row) {
    if (source->is_trans)
        return ERROR_TRANSPOSED;

    if (!source->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (last_row > source->size[0] || (first_row >= last_row))
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    source->data_device->sliceRows(first_row, last_row, *(target->data_device));
    target->on_device = 1;
    target->on_host = 0;
    target->size[1] = source->size[1];
    target->size[0] = last_row - first_row;
    target->is_trans = 0;
    target->owns_data = 1;
    return 0;
}

extern int add_mult(cudanetmat* mat1, cudanetmat* mat2, float alpha, float beta) {
    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat1->data_device->add(*(mat2->data_device), alpha, beta);
    return 0;
}

extern int set_col_slice(cudanetmat* source, cudanetmat* target, unsigned int start, unsigned int end) {
    int height = target->size[0];
    int width = target->size[1];

    if ((end - start) != source->size[1] || source->size[0] != height || start >= end || end > width)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    source->data_device->copy(*(target->data_device), 0, source->size[0], 0, source->size[1], 0, start);

    return 0;
}

extern int set_row_slice(cudanetmat* source, cudanetmat* target, unsigned int start, unsigned int end) {
    int height = target->size[0];
    int width = target->size[1];

    if ((end - start) != source->size[0] || source->size[1] != width || start >= end || end > height)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    source->data_device->copy(*(target->data_device), 0, source->size[0], 0, source->size[1], start, 0);

    return 0;
}

extern int assign_col_slice(cudanetmat* source, unsigned int start, unsigned int end, float val) {
    int height = source->size[0];
    int width = source->size[1];
    if (start >= end || end > width)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    source->data_device->assignSlice(0, height, start, end, val);

    return 0;
}

extern int assign_row_slice(cudanetmat* source, unsigned int start, unsigned int end, float val) {
    int height = source->size[0];
    int width = source->size[1];
    if (start >= end || end > height)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    source->data_device->assignSlice(start, end, 0, width, val);

    return 0;
}

extern int apply_pow_matrix(cudanetmat* mat, cudanetmat* pow, cudanetmat* target) {

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (mat->size[0] != pow->size[0] || mat->size[1] != pow->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->data_device->applyBinary(NVMatrixBinaryOps::Power(), *(pow->data_device), *(target->data_device));
    return 0;
}

extern int print_devmat(cudanetmat* mat) {
    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    mat->data_device->print(0, mat->data_device->getNumRows(), 0, mat->data_device->getNumCols());
    printf("stride: %d ld: %d, fd:%d\n", mat->data_device->getStride(), mat->data_device->getLeadingDim(), mat->data_device->getFollowingDim());
    return 0;
}
// extern int apply_pow_matrix(cudanetmat* mat, cudanetmat* pow, cudanetmat* target) {
//     int errcheck = elementwise_check2(mat, target);
//     mat->data_device->func(*(target->data_device));
// }
extern int reciprocal(cudanetmat* mat, cudanetmat* target) {
    int errcheck = elementwise_check2(mat, target);
    if (errcheck !=0) return errcheck;
    mat->data_device->apply(NVMatrixOps::Reciprocal(), *(target->data_device));
    return 0;
}

extern int free_device_memory(cudanetmat* mat) {
    if (mat->owns_data && mat->on_device) {
        delete mat->data_device;
        mat->data_device = NULL;
        mat->on_device = 0;

    }

    return 0;
}

extern float euclid_norm(cudanetmat* mat, int* err_code) {
    if (!mat->on_device) {
        *err_code = ERROR_NOT_ON_DEVICE;    
        return -1.;
    }

    float res = mat->data_device->norm();
    *err_code = 0;
    return res;
}

extern float manhattan_norm(cudanetmat* mat, int* err_code) {
    if (!mat->on_device) {
        *err_code = ERROR_NOT_ON_DEVICE;    
        return -1.;
    }

    float res = mat->data_device->sumabs();
    *err_code = 0;
    return res;
}

extern int less_than(cudanetmat* mat1, cudanetmat* mat2, cudanetmat* target) {
    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat1->data_device->applyBinary(NVMatrixBinaryOps::SmallerThan(), *(mat2->data_device), *(target->data_device));
    return 0;
}

extern int less_than_scalar(cudanetmat* mat, float val, cudanetmat* target) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->data_device->apply(NVMatrixOps::SmallerThanScalar(val), *(target->data_device));
    return 0;
}

extern int greater_than(cudanetmat* mat1, cudanetmat* mat2, cudanetmat* target) {
    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat1->data_device->applyBinary(NVMatrixBinaryOps::BiggerThan(), *(mat2->data_device), *(target->data_device));
    return 0;
}

extern int greater_than_scalar(cudanetmat* mat, float val, cudanetmat* target) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->data_device->apply(NVMatrixOps::BiggerThanScalar(val), *(target->data_device));
    return 0;
}

extern int equals(cudanetmat* mat1, cudanetmat* mat2, cudanetmat* target) {
    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat1->data_device->applyBinary(NVMatrixBinaryOps::Equals(), *(mat2->data_device), *(target->data_device));
    return 0;
}

extern int equals_scalar(cudanetmat* mat, float val, cudanetmat* target) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->data_device->apply(NVMatrixOps::EqualsScalar(val), *(target->data_device));
    return 0;
}


extern int where(cudanetmat* condition_mat, cudanetmat* if_mat, cudanetmat* else_mat, cudanetmat* target) {
    if (!condition_mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (condition_mat->size[0] != target->size[0] || condition_mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (condition_mat->size[0] != if_mat->size[0] || condition_mat->size[1] != if_mat->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
        
    if (condition_mat->size[0] != else_mat->size[0] || condition_mat->size[1] != else_mat->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    condition_mat->data_device->applyTernary(NVMatrixTernaryOps::Where(), *(if_mat->data_device), *(else_mat->data_device), *(target->data_device));
    return 0;
}

extern int minimum(cudanetmat* mat1, cudanetmat* mat2, cudanetmat* target) {
    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat1->data_device->applyBinary(NVMatrixBinaryOps::Minimum(), *(mat2->data_device), *(target->data_device));
    return 0;
}

extern int minimum_scalar(cudanetmat* mat, float val, cudanetmat* target) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->data_device->apply(NVMatrixOps::MinWithScalar(val), *(target->data_device));
    return 0;
}

extern int maximum(cudanetmat* mat1, cudanetmat* mat2, cudanetmat* target) {
    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat1->data_device->applyBinary(NVMatrixBinaryOps::Maximum(), *(mat2->data_device), *(target->data_device));
    return 0;
}

extern int maximum_scalar(cudanetmat* mat, float val, cudanetmat* target) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->data_device->apply(NVMatrixOps::MaxWithScalar(val), *(target->data_device));
    return 0;
}

extern int reshape(cudanetmat* mat, unsigned int m, unsigned int n) {
    if (mat->size[0] * mat->size[1] != m * n)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (mat->on_device)
        mat->data_device->resize(m,n);

    mat->size[0] = m;
    mat->size[1] = n;

    return 0;
}


extern int add_col_vec(cudanetmat* mat, cudanetmat* vec, cudanetmat* target) {
    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->data_device->addVector(*(vec->data_device), *(target->data_device));
    return 0;
}

extern int add_col_mult(cudanetmat* mat, cudanetmat* vec, cudanetmat* target, float mult) {
    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->data_device->addVector(*(vec->data_device), mult, *(target->data_device));
    return 0;
}

extern int add_row_vec(cudanetmat* mat, cudanetmat* vec, cudanetmat* target) {
    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->data_device->addVector(*(vec->data_device), *(target->data_device));
    return 0;
}


extern int mult_by_col_vec(cudanetmat* mat, cudanetmat* vec, cudanetmat* target) {
    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->data_device->eltwiseMultByVector(*(vec->data_device), *(target->data_device));

    return 0;
}

extern int mult_by_row_vec(cudanetmat* mat, cudanetmat* vec, cudanetmat* target) {
    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->data_device->eltwiseMultByVector(*(vec->data_device), *(target->data_device));

    return 0;
}

extern int divide_by_col_vec(cudanetmat* mat, cudanetmat* vec, cudanetmat* target) {
    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->data_device->eltwiseDivideByVector(*(vec->data_device), *(target->data_device));

    return 0;
}

extern int divide_by_row_vec(cudanetmat* mat, cudanetmat* vec, cudanetmat* target) {
    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->data_device->eltwiseDivideByVector(*(vec->data_device), *(target->data_device));

    return 0;
}

extern int max_by_axis(cudanetmat* mat, cudanetmat* target, int axis) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;
    if (axis == -1) {
        if (target->size[0] != 1 || target->size[1] != 1)
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        target->data_device->assign(mat->data_device->max());
    } else if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->max(0, *(target->data_device));
    } else {
        if (target->size[1] != 1 || target->size[0] != mat->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->max(1, *(target->data_device));
    }
    return 0;
}

extern int min_by_axis(cudanetmat* mat, cudanetmat* target, int axis) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == -1) {
        if (target->size[0] != 1 || target->size[1] != 1)
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        target->data_device->assign(mat->data_device->min());
    } else if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->min(0, *(target->data_device));
    } else {
        if (target->size[1] != 1 || target->size[0] != mat->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->min(1, *(target->data_device));
    }
    return 0;
}


extern int sum(cudanetmat* mat, cudanetmat* target, int axis) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == -1) {
        if (target->size[0] != 1 || target->size[1] != 1)
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        target->data_device->assign(mat->data_device->sum());
    } else if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->sum(0, *(target->data_device));
    } else {
        if (target->size[1] != 1 || target->size[0] != mat->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->sum(1, *(target->data_device));
    }
    return 0;
}


extern int sumsq(cudanetmat* mat, cudanetmat* target, int axis) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == -1) {
        if (target->size[0] != 1 || target->size[1] != 1)
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        NVMatrix tmp;
        mat->data_device->sumOfSquares(0, tmp);
        target->data_device->assign(tmp.sum());
    } else if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->sumOfSquares(0, *(target->data_device));
    } else {
        if (target->size[1] != 1 || target->size[0] != mat->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->sumOfSquares(1, *(target->data_device));
    }
    return 0;
}

extern int mean(cudanetmat* mat, cudanetmat* target, int axis) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == -1) {
        if (target->size[0] != 1 || target->size[1] != 1)
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        target->data_device->assign(mat->data_device->mean());
    } else if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->mean(0, *(target->data_device));
    } else {
        if (target->size[1] != 1 || target->size[0] != mat->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->mean(1, *(target->data_device));
    }
    return 0;
}

extern int var(cudanetmat* mat, cudanetmat* mean, cudanetmat* target, int axis) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == -1) {
        if (target->size[0] != 1 || target->size[1] != 1)
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        return ERROR_UNSUPPORTED;
    } else if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->var(0, *(mean->data_device), *(target->data_device));
    } else {
        if (target->size[1] != 1 || target->size[0] != mat->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->var(1, *(mean->data_device), *(target->data_device));
    }
    return 0;
}

extern int mean_norm(cudanetmat* mat, cudanetmat* target, int axis) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (axis == -1) {
        float mval = mat->data_device->mean();
        mat->data_device->addScalar(-mval, *(target->data_device));
    } else if (axis == 0 || axis == 1) {
        NVMatrix mvals;
        mat->data_device->mean(axis, mvals);
        mat->data_device->addVector(mvals, -1.0, *(target->data_device));
    } else {
        return ERROR_UNSUPPORTED;
    }
    return 0;
}

extern int argmax_by_axis(cudanetmat* mat, cudanetmat* target, int axis) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->argmax(0, *(target->data_device));
    } else {
        if (target->size[1] != 1 || target->size[0] != mat->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->argmax(1, *(target->data_device));
    }

    return 0;
}

extern int argmin_by_axis(cudanetmat* mat, cudanetmat* target, int axis) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->argmin(0, *(target->data_device));
    } else {
        if (target->size[1] != 1 || target->size[0] != mat->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
        mat->data_device->argmin(1, *(target->data_device));
    }

    return 0;
}

extern int copy_transpose(cudanetmat* source, cudanetmat* target) {
    if (source->size[0] != target->size[1] || source->size[1] != target->size[0])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    source->data_device->transpose(*(target->data_device));
    return 0;
}

extern int xcov(cudanetmat* X, cudanetmat* Y, cudanetmat* covMat, int normX, int normY, float normAll) {
    if (!X->on_device || !Y->on_device || !covMat->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (X->is_trans || Y->is_trans || covMat->is_trans)
        return ERROR_TRANSPOSED;

    if (get_nonleading_dimension(Y) != get_nonleading_dimension(X) ||
        get_leading_dimension(X) != get_leading_dimension(covMat) ||
        get_leading_dimension(Y) != get_nonleading_dimension(covMat)) {
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    }

    // Mean normalize each input matrix along major axis (for _cudanet, this is along 1) matrices are K x N
    // Xmean and Ymean are K-dim row vectors
    NVMatrix Xmean, Ymean;
    X->data_device->mean(1, Xmean);
    Y->data_device->mean(1, Ymean);

    // Now normalize in each
    NVMatrix Xnorm, Ynorm;
    X->data_device->addVector(Xmean, -1*normX, Xnorm);
    Y->data_device->addVector(Ymean, -1*normY, Ynorm);

    // Now calc the norm into covMat
    covMat->data_device->addProductRM(Xnorm, Ynorm, 0, 1/normAll, 0 /* trans of X */, 1 /* non-trans of Y*/);
    return 0;
}

extern unsigned long int get_gpu_pointer(cudanetmat* source) {
    return (unsigned long int) source->data_device->getDevData();
}

extern PyObject* get_gpu_pythonbuf(cudanetmat* source) {
    PyObject* py_buf = PyBuffer_FromReadWriteMemory((void *) (source->data_device->getDevData()), source->data_device->getNumElements() * sizeof(float));
    Py_INCREF(py_buf);
    return py_buf;
}

extern int multi_ranked_error(cudanetmat* probs, cudanetmat* labels, cudanetmat *labellogprob, cudanetmat* top1probs, cudanetmat* topkprobs, int topk) {
    NVMatrix _maxProbs;
    probs->data_device->max(0, _maxProbs);
    computeMultiSoftmaxCost(*(labels->data_device), *(probs->data_device), _maxProbs, *(labellogprob->data_device), *(top1probs->data_device), *(topkprobs->data_device), topk);
    return 0;
}

// If axis == 0, then mat is K x N where K is number of outputs, N is number of examples
// If axis == 1, then mat is N x K where K is number of outputs, N is number of examples
// Cudanet convention is axis = 0, so 
extern int softmax(cudanetmat* mat, cudanetmat* target, int axis) {
    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans || target->is_trans)
        return ERROR_TRANSPOSED;
    NVMatrix _max, _sum;
    NVMatrix& input = *(mat->data_device);
    NVMatrix& tgt = *(target->data_device);
    input.max(axis, _max);
    input.addVector(_max, -1, tgt);
    tgt.apply(NVMatrixOps::Exp());
    tgt.sum(axis, _sum);
    tgt.eltwiseDivideByVector(_sum);

    return 0;
}

// acts, actsGrad, and target are all numOut x BatchSize

extern int softmax_grad(cudanetmat* acts, cudanetmat* actsGrad, cudanetmat* target) {
    if (!acts->on_device || !actsGrad->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (acts->is_trans || actsGrad->is_trans || target->is_trans)
        return ERROR_TRANSPOSED;

    int errcheck = elementwise_check3(acts, actsGrad, target);
    if (errcheck !=0) return errcheck;
    acts->data_device->transpose(true);
    actsGrad->data_device->transpose(true);
    target->data_device->transpose(true);
    //Change assertion in computeSoftmaxgrad to just ensure that acts and actsGrad are same
    computeSoftmaxGrad(*(acts->data_device), *(actsGrad->data_device), *(target->data_device), 0, 1);
    acts->data_device->transpose(false);
    actsGrad->data_device->transpose(false);
    target->data_device->transpose(false);
    return 0;
}

// labels and outputs are numOut x BatchSize, target is 1 x BatchSize
extern int crossent_cost(cudanetmat* labels, cudanetmat* outputs, cudanetmat* target) {
    if (!labels->on_device || !outputs->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (labels->is_trans || outputs->is_trans || target->is_trans)
        return ERROR_TRANSPOSED;

    int errcheck = elementwise_check2(labels, outputs);
    if (errcheck !=0) return errcheck;

    NVMatrix correctProbs_out; // This gets resized in cost call
    computeCrossEntCost(*(labels->data_device), *(outputs->data_device), *(target->data_device), correctProbs_out);
    return 0;
}

extern int crossent_cost_grad(cudanetmat* labels, cudanetmat* outputs, cudanetmat* target) {
    if (!labels->on_device || !outputs->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (labels->is_trans || outputs->is_trans || target->is_trans)
        return ERROR_TRANSPOSED;

    int errcheck = elementwise_check2(labels, outputs);
    if (errcheck !=0) return errcheck;

    computeCrossEntGrad(*(labels->data_device), *(outputs->data_device), *(target->data_device), 0, 1);
    return 0;
}

extern int weight_norm_along_axis(cudanetmat* weights, cudanetmat* target, int axis, float norm) {
// checks if the l2 norm of weights along axis is greater than norm -- if so, scale so l2norm(weights) is norm

    if (!weights->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (weights->is_trans || target->is_trans)
        return ERROR_TRANSPOSED;

    if (axis!=0 && axis!=1) 
        return ERROR_UNSUPPORTED;

    NVMatrix normVect;
    weights->data_device->sumOfSquares(axis, normVect);
    normVect.apply(MaxWeightConstraintOperator(norm));
    weights->data_device->eltwiseMultByVector(normVect, *(target->data_device));
    return 0;
}

extern PyObject *test_make_tuple(int nval) {
    PyObject *t;

    t = Py_BuildValue("(iis)", nval, nval, "three");
    return t;
}


// These are still to do
// Weight column norm
// softmax grad
// cross entropy multi-class cost
//
}
