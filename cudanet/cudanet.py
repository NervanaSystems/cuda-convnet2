# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
import os

import ctypes as ct
import ctypes.util
from ctypes import pythonapi
import numpy as np

MAX_ONES = 1024*256

cudanet_lib_path = ct.util.find_library('cconv2_cudanet')
if cudanet_lib_path is None:
    raise OSError("Problems locating libcudanet shared library")
_cudanet = ct.cdll.LoadLibrary(cudanet_lib_path)


_cudanet.get_last_cuda_error.restype = ct.c_char_p
_cudanet.cublas_init.restype = ct.c_int
_cudanet.cublas_shutdown.restype = ct.c_int
_cudanet.cuda_set_device.restype = ct.c_int

_cudanet.init_empty.restype = ct.c_int
# _cudanet.reshape.restype = ct.c_int
_cudanet.copy_to_host.restype = ct.c_int
_cudanet.copy_from.restype = ct.c_int
_cudanet.set_host_mat.restype = ct.c_int
_cudanet.allocate_device_memory = ct.c_int
_cudanet.copy_to_device.restype = ct.c_int
_cudanet.copy_on_device.restype = ct.c_int
_cudanet.free_device_memory.restype = ct.c_int
_cudanet.add_elementwise.restype = ct.c_int
_cudanet.add_scalar.restype = ct.c_int
_cudanet.add_vector.restype = ct.c_int
_cudanet.mat_vector_op.restype = ct.c_int
_cudanet.assign_scalar.restype = ct.c_int
_cudanet.subtract_elementwise.restype = ct.c_int
_cudanet.divide_elementwise.restype = ct.c_int
_cudanet.mult_elementwise.restype = ct.c_int
_cudanet.mult_by_scalar.restype = ct.c_int
_cudanet.sign.restype = ct.c_int
_cudanet.apply_sigmoid.restype = ct.c_int
_cudanet.apply_tanh.restype = ct.c_int
_cudanet.apply_soft_threshold.restype = ct.c_int
_cudanet.apply_abs.restype = ct.c_int
_cudanet.apply_log_1_plus_exp.restype = ct.c_int
_cudanet.apply_gamma.restype = ct.c_int
_cudanet.apply_lgamma.restype = ct.c_int
_cudanet.apply_log.restype = ct.c_int
_cudanet.apply_clip_range.restype = ct.c_int
_cudanet.apply_exp.restype = ct.c_int
_cudanet.apply_sqrt.restype = ct.c_int
_cudanet.apply_pow.restype = ct.c_int
_cudanet.apply_pow_matrix.restype = ct.c_int
_cudanet.reciprocal.restype = ct.c_int
_cudanet.convolution.restype = ct.c_int
_cudanet.print_devmat.restype = ct.c_int
_cudanet.get_col_slice_view.restype = ct.c_int
_cudanet.get_col_slice_copy.restype = ct.c_int
_cudanet.set_col_slice.restype = ct.c_int
_cudanet.get_row_slice_view.restype = ct.c_int
_cudanet.get_row_slice_copy.restype = ct.c_int
_cudanet.set_row_slice.restype = ct.c_int
_cudanet.assign_col_slice.restype = ct.c_int
_cudanet.assign_row_slice.restype = ct.c_int

_cudanet.euclid_norm.restype = ct.c_float
_cudanet.manhattan_norm.restype = ct.c_float
_cudanet.vdot.restype = ct.c_float
_cudanet.dot.restype = ct.c_int

_cudanet.less_than.restype = ct.c_int
_cudanet.less_than_scalar.restype = ct.c_int
_cudanet.greater_than.restype = ct.c_int
_cudanet.greater_than_scalar.restype = ct.c_int
_cudanet.equals.restype = ct.c_int
_cudanet.equals_scalar.restype = ct.c_int
_cudanet.minimum.restype = ct.c_int
_cudanet.minimum_scalar.restype = ct.c_int
_cudanet.maximum.restype = ct.c_int
_cudanet.maximum_scalar.restype = ct.c_int
_cudanet.reshape.restype = ct.c_int
_cudanet.add_col_vec.restype = ct.c_int
_cudanet.add_col_mult.restype = ct.c_int
_cudanet.add_row_vec.restype = ct.c_int
_cudanet.mult_by_col_vec.restype = ct.c_int
_cudanet.mult_by_row_vec.restype = ct.c_int
_cudanet.divide_by_col_vec.restype = ct.c_int
_cudanet.divide_by_row_vec.restype = ct.c_int
_cudanet.max_by_axis.restype = ct.c_int
_cudanet.min_by_axis.restype = ct.c_int
_cudanet.sum.restype = ct.c_int
_cudanet.sumsq.restype = ct.c_int
_cudanet.mean.restype = ct.c_int
_cudanet.convolution_back_errors.restype = ct.c_int
_cudanet.convolution_back_weights.restype = ct.c_int
_cudanet.copy_transpose.restype = ct.c_int

_cudanet.max_pool.restype = ct.c_int
_cudanet.max_pool_undo.restype = ct.c_int
_cudanet.avg_pool.restype = ct.c_int
_cudanet.avg_pool_undo.restype = ct.c_int
_cudanet.l2_pool.restype = ct.c_int
_cudanet.l2_pool_undo.restype = ct.c_int
_cudanet.unpool_forward.restype = ct.c_int
_cudanet.unpool_backward.restype = ct.c_int

_cudanet.adadelta_update.restype = ct.c_int
_cudanet.xcov.restype = ct.c_int
_cudanet.mean_norm.restype = ct.c_int
_cudanet.crossmap_response_norm.restype = ct.c_int
_cudanet.crossmap_response_norm_undo.restype = ct.c_int
_cudanet.local_contrast_norm.restype = ct.c_int
_cudanet.local_contrast_norm_undo.restype = ct.c_int
_cudanet.get_gpu_pointer.restype = ct.c_ulong
_cudanet.get_device_id.restype = ct.c_int
_cudanet.set_device_id.restype = None
_cudanet.get_peer_access.restype = ct.c_int
_cudanet.get_data_device_id.restype = ct.c_int
_cudanet.randomize_gaussian.restype = ct.c_int
_cudanet.randomize_uniform.restype = ct.c_int
_cudanet.randomize_binary.restype = ct.c_int
_cudanet.add_noise_gaussian.restype = ct.c_int
_cudanet.add_noise_uniform.restype = ct.c_int
_cudanet.randomize_uniform_thresh.restype = ct.c_int
_cudanet.init_random.restype = None
_cudanet.init_random_no_seed.restype = None
_cudanet.destroy_random.restype = None
_cudanet.sync_stream.restype = None
_cudanet.softmax.restype = ct.c_int
_cudanet.softmax_grad.restype = ct.c_int
_cudanet.crossent_cost.restype = ct.c_int
_cudanet.crossent_cost_grad.restype = ct.c_int
_cudanet.get_gpu_pythonbuf.restype = ct.py_object
_cudanet.multi_ranked_error.restype = ct.c_int

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    def newFunc(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc



class CUDANetException(Exception):
    pass

def get_last_cuda_error():
    return str(_cudanet.get_last_cuda_error())

def sync_stream():
    """
    Sets the current deviceid context
    """
    _cudanet.sync_stream()

def set_device_id(d):
    """
    Sets the current deviceid context
    """
    _cudanet.set_device_id(ct.c_int(d))

def get_device_id():
    """
    Returns the current deviceid context
    """
    return _cudanet.get_device_id()

def get_num_devices():
    """
    Returns the current deviceid context
    """

    err_code = ct.c_int(0)
    numdev = _cudanet.get_num_devices(ct.byref(err_code))
    if (err_code):
        generate_exception(err_code)
    return numdev


def get_peer_access(src, dest):
    """
    Returns whether deviceid src to deviceid dest access available
    """
    return _cudanet.set_peer_access(ct.c_int(src), ct.c_int(dest))

def generate_exception(err_code):
    """
    Return a CUDANetException object based on the error code err_code.
    """
    if err_code == -1:
        return CUDANetException("Incompatible matrix dimensions.")
    elif err_code == -2:
        return CUDANetException("CUBLAS error.")
    elif err_code == -3:
        return CUDANetException("CUDA error: " + get_last_cuda_error())
    elif err_code == -4:
        return CUDANetException("Operation not supported on views.")
    elif err_code == -5:
        return CUDANetException("Operation not supported on transposed matrices.")
    elif err_code == -6:
        return CUDANetException("Invalid value")
    elif err_code == -7:
        return CUDANetException("Incompatible transposedness.")
    elif err_code == -8:
        return CUDANetException("Matrix is not in device memory.")
    elif err_code == -9:
        return CUDANetException("Operation not supported.")
    elif err_code == -10:
        return CUDANetException("Convolutional dimensions incorrect")
    elif err_code == -11:
        return CUDANetException("Convolution Number of filters must be multiple of 16.")
    elif err_code == -12:
        return CUDANetException("Invalid axis type")
    elif err_code == -13:
        return CUDANetException("Randomizer not initialized")

class NVMat(ct.Structure):
	pass
class HostMat(ct.Structure):
	pass

class _PY_BUFFER(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.py_object),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("smalltable", ctypes.c_ssize_t * 2),
        ("internal", ctypes.c_void_p)
    ]

class cudanetmat(ct.Structure):
    _fields_ = [('data_host', ct.POINTER(HostMat)),
                ('data_device', ct.POINTER(NVMat)),
                ('on_device', ct.c_int),
                ('on_host', ct.c_int),
                ('size', ct.c_int * 2),
                ('is_trans', ct.c_int),
                ('owns_data', ct.c_int)]

class rnd_struct(ct.Structure):
    _fields_ = [('dev_rnd_mults', ct.POINTER(ct.c_uint)), 
                ('dev_rnd_words', ct.POINTER(ct.c_longlong))]


class TransposedCUDAMatrix(object):
    def __init__(self, mat):
        self.mat = cudanetmat()
        ct.memmove(ct.pointer(self.mat), ct.pointer(mat), ct.sizeof(self.mat))
        self.mat.is_trans = 1
        self.p_mat = ct.pointer(self.mat)

class CUDAMatrix(object):
    """
    A CUDAMatrix object represents a matrix of single precision floating point
    numbers on a GPU.
    """

    def __init__(self, array, copy_to_device = True, copy_on_host = True):
        """
        Initializes a new matrix object in one of two ways. If array is a numpy
        ndarray, memory for a matrix with the same dimensions is allocated on
        the GPU. If the copy_to_device flag is set to True, the GPU matrix is
        initialized with the given ndarray. If the copy_on_host flag is set to
        True, a copy of the matrix will be created in host memory even if the
        matrix is of the correct type (float32, Fortran-contiguous order).
        If array is not an ndarray, it must be a cudanetmat structure (typically
        the user will never use this way of calling __init__).
        """

        if type(array) in [np.ndarray, np.memmap]:
            # Convert array to float32 in FORTRAN order
            # array = reformat(array, copy = copy_on_host)

            # Initialize as a ndarray-tied matrix.
            self.mat = cudanetmat()
            self.size = self.mat.size
            self.p_mat = ct.pointer(self.mat)
            self.numpy_array = array

            _cudanet.init_from_array(self.p_mat, array.ctypes.data_as(ct.POINTER(ct.c_float)), ct.c_int(array.shape[0]), ct.c_int(array.shape[1]))
            if copy_to_device:
                err_code = _cudanet.copy_to_device(self.p_mat)
                if err_code:
                    raise generate_exception(err_code)

        else:
            # Initialize based on existing cudamat structure.
            mat = array
            self.mat = mat
            self.p_mat = ct.pointer(self.mat)
            self.size = self.mat.size

        self.T = TransposedCUDAMatrix(self.mat)

        # Keep a reference to free device memory in case of a crash.
        self.__free_device_memory = _cudanet.free_device_memory

    def __del__(self):
        try:
            if 'p_mat' in self.__dict__:
                err_code = self.__free_device_memory(self.p_mat)
                if err_code:
                    raise generate_exception(err_code)
        except AttributeError:
            pass

    @staticmethod
    def init_random(seed = None):
        pass
    #     """
    #     Initialize and seed the random number generator.
    #     """

    #     NUM_RND_STREAMS = 96*128
    #     CUDAMatrix.rndInitialized = 1
    #     CUDAMatrix.rnd_state = rnd_struct()
    #     CUDAMatrix.rnd_state_p = ct.pointer(CUDAMatrix.rnd_state)

    #     cudamat_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rnd_multipliers_32bit.txt')

    #     err_code = _cudanet.init_random(CUDAMatrix.rnd_state_p, ct.c_int(seed), cudamat_path)
    #     if err_code:
    #         raise generate_exception(err_code)

    @property
    def shape(self):
        return (self.mat.size[0], self.mat.size[1])

    def reshape(self, shape):
        """
        Reshapes self to have the given shape. The number of elements cannot
        change as this only changes how the contents are interpreted.
        """

        m = ct.c_uint(shape[0])
        n = ct.c_uint(shape[1])

        mat = cudanetmat()

        err_code = _cudanet.reshape(self.p_mat, ct.pointer(mat), m, n)

        if err_code:
            raise generate_exception(err_code)

        new_mat = CUDAMatrix(mat)

        return new_mat

    def asarray(self):
        """
        Copies the matrix to an ndarray on the CPU and returns it.
        """

        self.copy_to_host()

        return self.numpy_array

    def get_gpu_pointer(self):
        """
        Return the gpu pointer
        """
        return _cudanet.get_gpu_pointer(self.p_mat)

    def get_data_device_id(self):
        """
        Return the gpu pointer
        """
        dev_id = _cudanet.get_data_device_id(self.p_mat)

        # Error codes only for negative device ids
        if (dev_id < 0):
            raise generate_exception(dev_id)
        else:
            return dev_id

    def copy_to_device(self):
        """
        Copy the matrix to the GPU.
        """

        err_code = _cudanet.copy_to_device(self.p_mat)
        if err_code:
            raise generate_exception(err_code)

    def copy_to_host(self):
        """
        Copy the matrix to the CPU.
        """

        if not self.mat.on_host:
            # allocate host storage if necessary
            m = self.mat.size[0]
            n = self.mat.size[1]
            self.numpy_array = np.empty((m, n), dtype=np.float32, order = 'C')
            _cudanet.set_host_mat(self.p_mat, self.numpy_array.ctypes.data_as(ct.POINTER(ct.c_float)))

            self.mat.on_host = 1

        err_code = _cudanet.copy_to_host(self.p_mat)
        if err_code:
            raise generate_exception(err_code)

    def get_gpu_pythonbuf(self):
        print "about to return the pybuf"
        return _cudanet.get_gpu_pythonbuf(self.p_mat)

    def _memoryView(self):
        SHAPE = ctypes.c_ssize_t * 1
        STRIDES = ctypes.c_ssize_t * 1

        pybuffer = _PY_BUFFER()
        pybuffer.buf = self.get_gpu_pointer()
        pybuffer.obj = ctypes.py_object()
        pybuffer.len = self.mat.size[0] * self.mat.size[1]
        pybuffer.itemsize = 4
        pybuffer.readonly = 0
        pybuffer.ndim = 1
        pybuffer.format = 'f'
        pybuffer.shape = SHAPE(self.mat.size[0] * self.mat.size[1])
        pybuffer.strides = STRIDES(1)
        pybuffer.suboffsets = ctypes.POINTER(ctypes.c_ssize_t)()
        pybuffer.smalltable[0] = 0
        pybuffer.smalltable[1] = 0
        pybuffer.internal = ctypes.c_void_p()

        pythonapi.PyMemoryView_FromBuffer.argtypes = [ctypes.POINTER(_PY_BUFFER)]
        pythonapi.PyMemoryView_FromBuffer.restype = ctypes.py_object

        return pythonapi.PyMemoryView_FromBuffer(ctypes.byref(pybuffer))

    def copy(self, include_host = False):
        """
        Create a copy of the matrix on GPU. If include_host is True, also
        creates a copy of the matrix on CPU if there was any.
        """

        new_mat = empty(self.shape).assign(self)

        if include_host and self.mat.on_host:
            new_mat.numpy_array = self.numpy_array.copy()
            _cudanet.set_host_mat(new_mat.p_mat, new_mat.numpy_array.ctypes.data_as(ct.POINTER(ct.c_float)))
            new_mat.mat.on_host = 1

        return new_mat

    def copy_from(self, src, is_trans=False):
        """
        Copy the source matrix from the host.
        """
        _cudanet.copy_from(self.p_mat, src.ctypes.data_as(ct.POINTER(ct.c_float)), ct.c_bool(is_trans))

    def assign(self, val):
        """Assign val to self, where val can be a scalar or a CUDAMatrix
        with the same dimensions as self. """

        if isinstance(val, CUDAMatrix):
            err_code = _cudanet.copy_on_device(val.p_mat, self.p_mat)
        elif isinstance(val, (np.int32, np.float32, int, float)):
            err_code = _cudanet.assign_scalar(self.p_mat, ct.c_float(val))
        else:
            raise ValueError("Assigned value must be of type CUDAMatrix, int, or float.")
            
        if err_code:
            raise generate_exception(err_code)

        return self

    def set_host_mat(self, newbuf):
        """
        For changing the host matrix associated with this object to newbuf
        """
        if not isinstance(newbuf, np.ndarray):
            raise ValueError("Assigned value must be a numpy array")
        _cudanet.set_host_mat(self.p_mat, newbuf.ctypes.data_as(ct.POINTER(ct.c_float)))
        self.numpy_array = newbuf

    def free_device_memory(self):
        """
        Free memory used up by the matrix on the GPU.
        """

        err_code = _cudanet.free_device_memory(self.p_mat)
        if err_code:
            raise generate_exception(err_code)

    def mult_by_scalar(self, alpha, target = None):
        """
        Multiply the matrix by a scalar.
        """

        if not target:
            target = self

        err_code = _cudanet.mult_by_scalar(self.p_mat, ct.c_float(alpha), target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target
    def dot(self, mat2, target = None):
        """
        Multiply the matrix by mat2 from the right.
        """

        return dot(self, mat2, target)

    def add_dot(self, m1, m2, mult = 1., beta = 1.):
        """
        Add the dot product of m1 and m2 to the matrix, scaled by mult.
        Self is scaled by beta before adding anything.
        """

        err_code = _cudanet.dot(m1.p_mat, m2.p_mat, self.p_mat, ct.c_float(beta), ct.c_float(mult))
        if err_code:
            raise generate_exception(err_code)

        return self

    def subtract_dot(self, m1, m2, mult = 1., beta = 1.):
        """
        Subtract the dot product of m1 and m2 from the matrix, scaled by mult.
        Self is scaled by beta before subtracting anything.
        """
        
        return self.add_dot(m1, m2, mult = -1. * mult, beta = beta)
    def add_mult(self, mat2, alpha = 1.):
        """
        Add multiple of mat2 to the matrix.
        """

        err_code = _cudanet.add_mult(self.p_mat, mat2.p_mat, ct.c_float(alpha))
        if err_code:
            raise generate_exception(err_code)

        return self
    
    def subtract_mult(self, mat2, alpha = 1.):
        """
        Subtract a multiple of mat2 from the matrix.
        """

        err_code = _cudanet.add_mult(self.p_mat, mat2.p_mat, ct.c_float(-1. * alpha))
        if err_code:
            raise generate_exception(err_code)

        return self

    def add_vector(self, vect, scale=1.0, target=None):
        if not target:
            target = self
        err_code = _cudanet.add_vector(self.p_mat, vect.p_mat, ct.c_float(scale), target.p_mat)
        if err_code:
            raise generate_exception(err_code)
        return target

    def add(self, val, target = None):
        """Add val to self, where val can be a scalar or a CUDAMatrix with the
        same dimensions as self. """

        if not target:
            target = self

        if isinstance(val, CUDAMatrix):
            if self.shape == val.shape:
                err_code = _cudanet.add_elementwise(self.p_mat, val.p_mat, target.p_mat)
            elif val.shape[0] == 1 or val.shape[1] == 1:
                err_code = _cudanet.add_vector(self.p_mat, val.p_mat, ct.c_float(1.0), target.p_mat)
        elif isinstance(val, (np.int32, np.float32, int, float)):
            err_code = _cudanet.add_scalar(self.p_mat, ct.c_float(val), target.p_mat)
        else:
            raise ValueError("Value must be of type CUDAMatrix, int, or float.")

        if err_code:
            raise generate_exception(err_code)

        return target

    def subtract(self, val, target = None):
        """Subtract val from self, where val can be a scalar or a CUDAMatrix with
        the same dimensions as self. """

        if not target:
            target = self

        if isinstance(val, CUDAMatrix):
            if self.shape == val.shape:
                err_code = _cudanet.subtract_elementwise(self.p_mat, val.p_mat, target.p_mat)
            elif val.shape[0] == 1 or val.shape[1] == 1:
                err_code = _cudanet.mat_vector_op(self.p_mat, val.p_mat, ct.c_float(1.0), target.p_mat, ct.c_char('s'))
        elif isinstance(val, (np.int32, np.float32, int, float)):
            err_code = _cudanet.add_scalar(self.p_mat, ct.c_float(-1*val), target.p_mat)
        else:
            raise ValueError("Value must be of type CUDAMatrix, int, or float.")

        if err_code:
            raise generate_exception(err_code)

        return target

    def divide(self, val, target = None):
        """Divide self by val, where val can be a scalar or a CUDAMatrix with the
        same dimensions as self. """

        if not target:
            target = self

        if isinstance(val, CUDAMatrix):
            if self.shape == val.shape:
                err_code = _cudanet.divide_elementwise(self.p_mat, val.p_mat, target.p_mat)
            elif val.shape[0] == 1 or val.shape[1] == 1:
                err_code = _cudanet.mat_vector_op(self.p_mat, val.p_mat, ct.c_float(1.0), target.p_mat, ct.c_char('d'))
        elif isinstance(val, (np.int32, np.float32, int, float)):
            err_code = _cudanet.divide_by_scalar(self.p_mat, ct.c_float(val), target.p_mat)
        else:
            raise ValueError("Value must be of type CUDAMatrix, int, or float.")

        if err_code:
            raise generate_exception(err_code)

        return target

    def mult(self, val, target = None):
        """Multiply self by val, where val can be a scalar or a CUDAMatrix with
        the same dimensions as self. """

        if not target:
            target = self

        if isinstance(val, CUDAMatrix):
            if self.shape == val.shape:
                err_code = _cudanet.mult_elementwise(self.p_mat, val.p_mat, target.p_mat)
            elif val.shape[0] == 1 or val.shape[1] == 1:
                err_code = _cudanet.mat_vector_op(self.p_mat, val.p_mat, ct.c_float(1.0), target.p_mat, ct.c_char('m'))
        elif isinstance(val, (np.int32, np.float32, int, float)):
            err_code = _cudanet.mult_by_scalar(self.p_mat, ct.c_float(val), target.p_mat)
        else:
            raise ValueError("Value must be of type CUDAMatrix, int, or float.")

        if err_code:
            raise generate_exception(err_code)

        return target

    def reciprocal(self, target = None):
        """Return the reciprocal """

        if not target:
            target = self

        err_code = _cudanet.reciprocal(self.p_mat, target.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target

    def slice_view(self, first_row, last_row, first_col, last_col, include_host = False):
        """
        Creates a view into a consecutive range of columns of an existing
        matrix on GPU. If include_host is set to True, also creates a view
        into the CPU copy of the matrix (i.e., the numpy_array).
        """
        mat = cudanetmat()

        err_code = _cudanet.get_slice_view(self.p_mat, ct.pointer(mat), ct.c_int(first_row), ct.c_int(last_row), ct.c_int(first_col), ct.c_int(last_col))

        if err_code:
            raise generate_exception(err_code)

        new_mat = CUDAMatrix(mat)

        try:
            new_mat.sliceof = self.sliceof
        except:
            new_mat.sliceof = self

        # reproduce the slice on the host as well (if requested)
        if include_host and self.mat.on_host:
            new_mat.numpy_array = self.numpy_array[first_row:last_row, first_col:last_col]
            _cudanet.set_host_mat(new_mat.p_mat, new_mat.numpy_array.ctypes.data_as(ct.POINTER(ct.c_float)))
            new_mat.mat.on_host = 1

        return new_mat

    def col_slice_view(self, first_col, last_col, include_host = False):
        return self.slice_view(0, self.mat.size[0], first_col, last_col, include_host)

    def row_slice_view(self, first_row, last_row, include_host = False):
        return self.slice_view(first_row, last_row, 0, self.mat.size[1], include_host)

    def slice(self, first_col, last_col, include_host = False):
        """
        Creates a view into a consecutive range of columns of an existing
        matrix on GPU. If include_host is set to True, also creates a view
        into the CPU copy of the matrix (i.e., the numpy_array).
        """
        mat = cudanetmat()

        if self.mat.size[0] == 1 or self.mat.size[1] == 1:
            err_code = _cudanet.get_vector_slice(self.p_mat, ct.pointer(mat), ct.c_int(first_col), ct.c_int(last_col))
        else:
            err_code = _cudanet.get_col_slice_view(self.p_mat, ct.pointer(mat), ct.c_int(first_col), ct.c_int(last_col))

        if err_code:
            raise generate_exception(err_code)

        new_mat = CUDAMatrix(mat)

        try:
            new_mat.sliceof = self.sliceof
        except:
            new_mat.sliceof = self

        # reproduce the slice on the host as well (if requested)        
        if include_host and self.mat.on_host:
            new_mat.numpy_array = self.numpy_array[:, first_col:last_col]
            _cudanet.set_host_mat(new_mat.p_mat, new_mat.numpy_array.ctypes.data_as(ct.POINTER(ct.c_float)))
            new_mat.mat.on_host = 1

        return new_mat

    def get_col_slice(self, start, end, target = None):
        """
        Get the columns with indices start through end. If target is not provided
        memory for a new matrix will be allocated.
        """
        height = self.shape[0]

        if not target:
            target = empty((height, end-start))

        err_code = _cudanet.get_col_slice_copy(self.p_mat, target.p_mat, ct.c_int(start), ct.c_int(end))
        if err_code:
            raise generate_exception(err_code)

        return target

    def set_col_slice(self, start, end, mat):
        """
        Assign the contents of mat to the columns with indices first_col
        through last_col.
        """
        if isinstance(mat, CUDAMatrix):
            err_code = _cudanet.set_col_slice(mat.p_mat, self.p_mat, ct.c_int(start), ct.c_int(end))
        else:
            err_code = _cudanet.assign_col_slice(self.p_mat, ct.c_int(start), ct.c_int(end), ct.c_float(mat))

        if err_code:
            raise generate_exception(err_code)

        return self

    def get_row_slice(self, start, end, target = None):
        """
        Get the rows with indices start through end. If target is not provided
        memory for a new matrix will be allocated.
        """
        width = self.shape[1]

        if not target:
            target = empty((end-start, width))

        err_code = _cudanet.get_row_slice_copy(self.p_mat, target.p_mat, ct.c_int(start), ct.c_int(end))
        if err_code:
            raise generate_exception(err_code)

        return target

    def set_row_slice(self, start, end, mat):
        """
        Assign the contents of mat to the rows with indices start through end.
        """
        if isinstance(mat, CUDAMatrix):
            err_code = _cudanet.set_row_slice(mat.p_mat, self.p_mat, ct.c_int(start), ct.c_int(end))
        else:
            err_code = _cudanet.assign_row_slice(self.p_mat, ct.c_int(start), ct.c_int(end), ct.c_float(mat))

        if err_code:
            raise generate_exception(err_code)

        return self

    @deprecated
    def div_by_scalar(self, alpha, target = None):
        """
        Divide the matrix by a scalar.
        """

        if not target:
            target = self

        err_code = _cudanet.divide_by_scalar(self.p_mat, ct.c_float(alpha), target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    @deprecated
    def add_scalar(self, alpha, target = None):
        """
        Increment the matrix by a scalar.
        """

        if not target:
            target = self

        err_code = _cudanet.add_scalar(self.p_mat, ct.c_float(alpha), target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def sum(self, axis, target = None, mult = 1.):
        """
        Sum the matrix along the given dimension, where 0 represents the leading
        dimension and 1 represents the non-leading dimension. If a target is
        not provided, a new vector is created for storing the result. The result
        is multiplied by the given factor mult (defaults to 1).
        """
        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n))
 
        elif axis == 1:
            if not target:
                target = empty((m, 1))
        elif axis == None:
            if not target:
                target = empty((1,1))
            axis = -1

        err_code = _cudanet.sum(self.p_mat, target.p_mat, ct.c_int(axis))

        if err_code:
            raise generate_exception(err_code)

        if (mult != 1. ):
            err_code = _cudanet.mult_by_scalar(target.p_mat, ct.c_float(mult), target.p_mat)
            if err_code:
                raise generate_exception(err_code)

        return target

    def sumsq(self, axis, target = None, mult = 1.):
        """
        Sum the matrix along the given dimension, where 0 represents the leading
        dimension and 1 represents the non-leading dimension. If a target is
        not provided, a new vector is created for storing the result. The result
        is multiplied by the given factor mult (defaults to 1).
        """
        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n))
        elif axis == 1:
            if not target:
                target = empty((m, 1))
        elif axis == None:
            if not target:
                target = empty((1,1))
            axis = -1

        err_code = _cudanet.sumsq(self.p_mat, target.p_mat, ct.c_int(axis))

        if err_code:
            raise generate_exception(err_code)

        if (mult != 1. ):
            err_code = _cudanet.mult_by_scalar(target.p_mat, ct.c_float(mult), target.p_mat)
            if err_code:
                raise generate_exception(err_code)

        return target

    def mean(self, axis, target = None):
        """
        Compute the mean of the matrix along the given dimension, where 0
        represents the leading dimension and 1 represents the non-leading
        dimension. If a target is not provided, a new vector is created for
        storing the result.
        """
        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n))
        elif axis == 1:
            if not target:
                target = empty((m, 1))
        elif axis == None:
            if not target:
                target = empty((1,1))
            axis = -1

        err_code =  _cudanet.mean(self.p_mat, target.p_mat, ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target

    def var(self, axis, mean, target = None):
        """
        Compute the variance of the matrix along the given dimension, where 0
        represents the leading dimension and 1 represents the non-leading
        dimension. If a target is not provided, a new vector is created for
        storing the result.
        """
        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n))
        elif axis == 1:
            if not target:
                target = empty((m, 1))
        else:
            raise ValueError("Axis must be provided (0 or 1)")

        err_code =  _cudanet.var(self.p_mat, mean.p_mat, target.p_mat, ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target

    def equals(self, val, target = None):
        """
        Perform the operation target = 1. * (self == val), where val can be a matrix or a scalar.
        """

        if not target:
            target = self

        if isinstance(val, (np.int32, np.float32, int, float)):
            err_code = _cudanet.equals_scalar(self.p_mat, ct.c_float(val), target.p_mat)
        else:
            err_code = _cudanet.equals(self.p_mat, val.p_mat, target.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target

    def less_than(self, val, target = None):
        """
        Perform the operation target = 1. * (self < val), where val can be a matrix or a scalar.
        """

        if not target:
            target = self

        if isinstance(val, (np.int32, np.float32, int, float)):
            err_code = _cudanet.less_than_scalar(self.p_mat, ct.c_float(val), target.p_mat)
        else:
            err_code = _cudanet.less_than(self.p_mat, val.p_mat, target.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target

    def greater_than(self, val, target = None):
        """
        Perform the operation target = 1. * (self > val), where val can be a matrix or a scalar.
        """

        if not target:
            target = self

        if isinstance(val, (np.int32, np.float32, int, float)):
            err_code = _cudanet.greater_than_scalar(self.p_mat, ct.c_float(val), target.p_mat)
        else:
            err_code = _cudanet.greater_than(self.p_mat, val.p_mat, target.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target

    def min(self, axis, target = None):
        """
        Find the minimum value along the given dimension, where 0 represents the
        leading dimension and 1 represents the non-leading dimension. If a target
        is not provided, a new vector is created for storing the result.
        """

        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n))
 
        elif axis == 1:
            if not target:
                target = empty((m, 1))
        elif axis == None:
            if not target:
                target = empty((1,1))
            axis = -1

        err_code =  _cudanet.min_by_axis(self.p_mat, target.p_mat, ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target

    def max(self, axis, target = None):
        """
        Find the maximum value along the given dimension, where 0 represents the
        leading dimension and 1 represents the non-leading dimension. If a target
        is not provided, a new vector is created for storing the result.
        """

        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n))
 
        elif axis == 1:
            if not target:
                target = empty((m, 1))
        elif axis == None:
            if not target:
                target = empty((1,1))
            axis = -1

        err_code =  _cudanet.max_by_axis(self.p_mat, target.p_mat, ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target

    def sum(self, axis, target = None):
        """
        Find the maximum value along the given dimension, where 0 represents the
        leading dimension and 1 represents the non-leading dimension. If a target
        is not provided, a new vector is created for storing the result.
        """

        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n))
 
        elif axis == 1:
            if not target:
                target = empty((m, 1))
        elif axis == None:
            if not target:
                target = empty((1,1))
            axis = -1

        err_code =  _cudanet.sum(self.p_mat, target.p_mat, ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target

    def mean_norm(self, axis, target = None):
        """
        Find the maximum value along the given dimension, where 0 represents the
        leading dimension and 1 represents the non-leading dimension. If a target
        is not provided, a new vector is created for storing the result.
        """

        m, n = self.shape

        if not target: 
            target = empty((m,n))

        err_code =  _cudanet.mean_norm(self.p_mat, target.p_mat, ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target

    def euclid_norm(self):
        """
        Returns the L2 norm of the matrix flattened to a vector.
        """
        err_code = ct.c_int(0)
        res = _cudanet.euclid_norm(self.p_mat, ct.byref(err_code))

        if err_code:
            raise generate_exception(err_code.value)

        return res

    def manhattan_norm(self):
        """
        Returns the L1 norm of the matrix flattened to a vector.
        """
        err_code = ct.c_int(0)
        res = _cudanet.manhattan_norm(self.p_mat, ct.byref(err_code))

        if err_code:
            raise generate_exception(err_code.value)

        return res

    def argmax(self, axis, target = None):
        """
        Find the index of the maximum value along the given dimension, where 0
        represents the leading dimension and 1 represents the non-leading
        dimension. If a target is not provided, a new vector is created for
        storing the result.
        """

        m, n = self.shape
        if axis == 0:
            if not target:
                target = empty((1, n))
 
        elif axis == 1:
            if not target:
                target = empty((m, 1))

        err_code =  _cudanet.argmax_by_axis(self.p_mat, target.p_mat, ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target

    def argmin(self, axis, target = None):
        """
        Find the index of the maximum value along the given dimension, where 0
        represents the leading dimension and 1 represents the non-leading
        dimension. If a target is not provided, a new vector is created for
        storing the result.
        """

        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n))
 
        elif axis == 1:
            if not target:
                target = empty((m, 1))

        err_code =  _cudanet.argmin_by_axis(self.p_mat, target.p_mat, ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target

    def randomize_gaussian(self, mean, stdev):
        """
        Fill in matrix with random values according to gaussian distribution with mean
        and stdev
        """
        err_code = _cudanet.randomize_gaussian(self.p_mat, ct.c_float(mean), ct.c_float(stdev));
        if err_code:
            raise generate_exception(err_code)

    def randomize_uniform(self):
        """
        Fill in matrix with random values according to uniform distribution
        between 0 and 1
        """
        err_code = _cudanet.randomize_uniform(self.p_mat);
        if err_code:
            raise generate_exception(err_code)

    def randomize_binary(self):
        """
        Fill in matrix with random values of {0,1} according to mask on uniform
        distribution
        """
        err_code = _cudanet.randomize_binary(self.p_mat);
        if err_code:
            raise generate_exception(err_code)

    def add_noise_gaussian(self, stdev):
        """
        Add noise to matrix according to gaussian distribution with standard deviation
        of stdev
        """
        err_code = _cudanet.add_noise_gaussian(self.p_mat, ct.c_float(stdev));
        if err_code:
            raise generate_exception(err_code)

    def add_noise_uniform(self, minRange, maxRange):
        """
        Add noise to matrix according to uniform distribution in [minRange, maxRange]
        """
        err_code = _cudanet.add_noise_uniform(self.p_mat, ct.c_float(minRange), ct.c_float(maxRange));
        if err_code:
            raise generate_exception(err_code)

    def randomize_uniform_thresh(self, keepthresh):
        """
        Fill in matrix with random values according to uniform distribution
        between 0 and 1, binarize to keep those below keepthresh, then scale by 1/keepthresh
        """
        err_code = _cudanet.randomize_uniform_thresh(self.p_mat, ct.c_float(keepthresh));
        if err_code:
            raise generate_exception(err_code)

    def set_trans(self, is_trans):
        """
        Set the transposedness flag to is_trans.
        """

        _cudanet.set_transpose(self.p_mat, ct.c_int(1 * is_trans))

    def print_devmat(self):
        """
        Set the transposedness flag to is_trans.
        """

        _cudanet.print_devmat(self.p_mat)
	
def empty(shape):
    """
    Creates and returns a new CUDAMatrix with the given shape.
    """

    mat = cudanetmat()
    err_code = _cudanet.init_empty(ct.pointer(mat), ct.c_int(shape[0]), ct.c_int(shape[1]))

    if err_code:
        raise generate_exception(err_code)

    return CUDAMatrix(mat)


def reformat(array, copy = True):
    """
    Returns array as a float32 array in FORTRAN order.
    If copy is set to False, the array will only be copied if it is not already
    in the correct format.
    """
    return np.array(array, dtype=np.float32, order='F', copy=copy)

def dot(m1, m2, target = None, beta = 0., alpha = 1.):
    """
    Find the dot product between m1 and m2 and store in target:
    target = beta*target + alpha*(m1 m2)
    If no target is given, it will be created automatically, but not
    initialized -- so beta should be left at its default value zero.
    """

    if not target:
        m = _cudanet.get_leading_dimension(m1.p_mat)
        n = _cudanet.get_nonleading_dimension(m2.p_mat)

        target = empty((m, n))

    err_code = _cudanet.dot(m1.p_mat, m2.p_mat, target.p_mat, ct.c_float(beta), ct.c_float(alpha))
    if err_code:
        raise generate_exception(err_code)

    return target

def vdot(m1, m2):
    """
    Compute the vector dot product of matrices m1 and m2.
    """

    err_code = ct.c_int(0)
    res = _cudanet.vdot(m1.p_mat, m2.p_mat, ct.byref(err_code))

    if err_code:
        raise generate_exception(err_code.value)

    return res

def sigmoid(mat, target = None):
    """
    Apply the logistic sigmoid to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudanet.apply_sigmoid(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def tanh(mat, target = None):
    """
    Apply the tanh to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudanet.apply_tanh(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def clip_range(mat, lower, upper, target = None):
    """
    Clip each element of the matrix to min of lower and max of upper

    """

    if not target:
        target = mat

    err_code = _cudanet.apply_clip_range(mat.p_mat, ct.c_float(lower), ct.c_float(upper), target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def soft_threshold(mat, alpha, target = None):
    """
    Apply the soft threshold function to each element of the matrix:

    mat = sign(mat) * max(0, abs(mat) - alpha)
    """

    if not target:
        target = mat

    err_code = _cudanet.apply_soft_threshold(mat.p_mat, ct.c_float(alpha), target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def abs(mat, target = None):
    """
    Apply abs to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudanet.apply_abs(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def log_1_plus_exp(mat, target = None):
    """
    Apply log(1+exp(x)) to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudanet.apply_log_1_plus_exp(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def maximum(mat, mat2, target = None):
    """
    Compute the element-wise max of mat and mat2
    """

    if not target:
        target = mat

    err_code = _cudanet.maximum(mat.p_mat, mat2.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def minimum(mat, mat2, target = None):
    """
    Compute the element-wise max of mat and mat2
    """

    if not target:
        target = mat

    err_code = _cudanet.minimum(mat.p_mat, mat2.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def maximum_scalar(mat, compval, target = None):
    """
    Compute the element-wise max of mat and compval (scalar)
    """

    if not target:
        target = mat

    err_code = _cudanet.maximum_scalar(mat.p_mat, ct.c_float(compval), target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def minimum_scalar(mat, compval, target = None):
    """
    Compute the minimum between mat and compval (scalar) elementwise
    """

    if not target:
        target = mat

    err_code = _cudanet.minimum_scalar(mat.p_mat, ct.c_float(compval), target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def log(mat, target = None):
    """
    Find the natural logarithm of each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudanet.apply_log(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def exp(mat, target = None):
    """
    Apply the exponential function to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudanet.apply_exp(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def sqrt(mat, target = None):
    """
    Compute the square root of each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudanet.apply_sqrt(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def pow(mat, p, target = None):
    """
    If p is a scalar, compute the 'p'th power of each element of the matrix mat,
    otherwise raise each element of the matrix mat to the power given by the
    corresponding element of the matrix p.
    """

    if not target:
        target = mat

    if isinstance(p, CUDAMatrix):
        err_code = _cudanet.apply_pow_matrix(mat.p_mat, p.p_mat, target.p_mat)
    elif isinstance(p, (np.int32, np.float32, int, float)):
        err_code = _cudanet.apply_pow(mat.p_mat, ct.c_float(p), target.p_mat)
    else:
        raise ValueError("Value must be of type CUDAMatrix, int, or float.")

    if err_code:
        raise generate_exception(err_code)

    return target

def cross_entropy(output, labels, target = None):
    """
    Compute the cross entropy between output and labels.  Can do multiple examples at a time.
    Dimensions of output and labels must match and the target collapses along the row axis
    """

    if not target:
        n = _cudanet.get_nonleading_dimension(output.p_mat)
        target = empty((1, n))

    err_code = _cudanet.cross_entropy(output.p_mat, labels.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def where(condition_mat, if_mat, else_mat, target = None):
    """
    For each element i, j, store if_math[i, j] in target[i,j] if
    condition_mat[i, j] is True, and else_mat[i, j] otherwise.
    """
    if not target:
        target = condition_mat

    err_code = _cudanet.where(condition_mat.p_mat, if_mat.p_mat, else_mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def max_pool(imgs, target, channels, sizeX, paddingStart, moduleStride, numModulesX):
    """
    Perform Max Pooling of kernel dimension sizeX on imgs and put result in target
    Images as  (CxHxW) Rows x (N) Columns in 'C' order 
    Target as  (KxPxQ) Rows x (N) Colums in 'C' order
    int imgSizeY, int numModulesX, int paddingStart, int moduleStride, 
    """

    err_code = _cudanet.max_pool(imgs.p_mat, target.p_mat, ct.c_int(channels),
                                 ct.c_int(sizeX), ct.c_int(paddingStart),
                                 ct.c_int(moduleStride), ct.c_int(numModulesX)) 
    if err_code:
        raise generate_exception(err_code)

    return target

def max_pool_undo(imgs, maxGrads, maxActs, target, sizeX, paddingStart, moduleStride, numModulesX):
    """
    Undo Max Pooling of kernel dimension sizeX on imgs and put result in target
    Images as  (CxHxW) Rows x (N) Columns in 'C' order 
    Target as  (KxPxQ) Rows x (N) Colums in 'C' order
    int imgSizeY, int numModulesX, int paddingStart, int moduleStride
    """

    err_code = _cudanet.max_pool_undo(imgs.p_mat, maxGrads.p_mat, maxActs.p_mat,
                                      target.p_mat, ct.c_int(sizeX),
                                      ct.c_int(paddingStart),
                                      ct.c_int(moduleStride),
                                      ct.c_int(numModulesX)) 
    if err_code:
        raise generate_exception(err_code)

    return target

def l2_pool(imgs, target, channels, sizeX, paddingStart, moduleStride, numModulesX):
    """
    Perform L2 Pooling of kernel dimension sizeX on imgs and put result in target
    Images as  (CxHxW) Rows x (N) Columns in 'C' order 
    Target as  (KxPxQ) Rows x (N) Colums in 'C' order
    int imgSizeY, int numModulesX, int paddingStart, int moduleStride, 
    """

    err_code = _cudanet.l2_pool(imgs.p_mat, target.p_mat, ct.c_int(channels),
                                 ct.c_int(sizeX), ct.c_int(paddingStart),
                                 ct.c_int(moduleStride), ct.c_int(numModulesX)) 
    if err_code:
        raise generate_exception(err_code)

    return target

def l2_pool_undo(imgs, l2Grads, l2Acts, target, sizeX, paddingStart, moduleStride, numModulesX):
    """
    Undo L2 Pooling of kernel dimension sizeX on imgs and put result in target
    Images as  (CxHxW) Rows x (N) Columns in 'C' order 
    Target as  (KxPxQ) Rows x (N) Colums in 'C' order
    int imgSizeY, int numModulesX, int paddingStart, int moduleStride
    """

    err_code = _cudanet.l2_pool_undo(imgs.p_mat, l2Grads.p_mat, l2Acts.p_mat,
                                      target.p_mat, ct.c_int(sizeX),
                                      ct.c_int(paddingStart),
                                      ct.c_int(moduleStride),
                                      ct.c_int(numModulesX)) 
    if err_code:
        raise generate_exception(err_code)

    return target

def avg_pool(imgs, target, channels, sizeX, paddingStart, moduleStride, numModulesX):
    """
    Perform Max Pooling of kernel dimension sizeX on imgs and put result in target
    Images as  (CxHxW) Rows x (N) Columns in 'C' order 
    Target as  (KxPxQ) Rows x (N) Colums in 'C' order
    int imgSizeY, int numModulesX, int paddingStart, int moduleStride, 
    """

    err_code = _cudanet.avg_pool(imgs.p_mat, target.p_mat, ct.c_int(channels),
                                 ct.c_int(sizeX), ct.c_int(paddingStart),
                                 ct.c_int(moduleStride), ct.c_int(numModulesX)) 
    if err_code:
        raise generate_exception(err_code)

    return target

def avg_pool_undo(avgGrads, target, sizeX, paddingStart, moduleStride, numModulesX, imgSizeX):
    """
    Undo Avg Pooling of kernel dimension sizeX on imgs and put result in target
    average Gradients as  (CxHxW) Rows x (N) Columns in 'C' order 
    Target as  (KxPxQ) Rows x (N) Colums in 'C' order
    int imgSizeY, int numModulesX, int paddingStart, int moduleStride, int numImgColors, int numGroups
    """

    err_code = _cudanet.avg_pool_undo(avgGrads.p_mat, target.p_mat,
                                      ct.c_int(sizeX), ct.c_int(paddingStart),
                                      ct.c_int(moduleStride),
                                      ct.c_int(numModulesX),
                                      ct.c_int(imgSizeX))
    if err_code:
        raise generate_exception(err_code)

    return target

def unpool_forward(smallMat, largeMat, channels, sizeX, smallX, largeX):
    """
    Undo Avg Pooling of kernel dimension sizeX on imgs and put result in target
    average Gradients as  (CxHxW) Rows x (N) Columns in 'C' order 
    Target as  (KxPxQ) Rows x (N) Colums in 'C' order
    int imgSizeY, int numModulesX, int paddingStart, int moduleStride, int numImgColors, int numGroups
    """

    err_code = _cudanet.unpool_forward(smallMat.p_mat, largeMat.p_mat,
                                       ct.c_int(channels), ct.c_int(sizeX),
                                       ct.c_int(smallX), ct.c_int(largeX))
    if err_code:
        raise generate_exception(err_code)

    return largeMat

def unpool_backward(largeMat, smallMat, channels, sizeX, smallX, largeX):
    """
    Undo Avg Pooling of kernel dimension sizeX on imgs and put result in target
    average Gradients as  (CxHxW) Rows x (N) Columns in 'C' order 
    Target as  (KxPxQ) Rows x (N) Colums in 'C' order
    int imgSizeY, int numModulesX, int paddingStart, int moduleStride, int numImgColors, int numGroups
    """

    err_code = _cudanet.unpool_backward(largeMat.p_mat, smallMat.p_mat,
                                       ct.c_int(channels), ct.c_int(sizeX),
                                       ct.c_int(smallX), ct.c_int(largeX))
    if err_code:
        raise generate_exception(err_code)

    return smallMat

def crossmap_response_norm(imgs, target, channels, sizeX, scale, power):
    """
    Perform response normalization across channels of kernel dimension sizeX on
    imgs and put result in target
    Images as  (CxHxW) Rows x (N) Columns in 'C' order 
    Target as  (KxPxQ) Rows x (N) Colums in 'C' order
    int channels, sizeX, float scale, power
    """

    err_code = _cudanet.crossmap_response_norm(imgs.p_mat, target.p_mat,
                                               ct.c_int(channels),
                                               ct.c_int(sizeX),
                                               ct.c_float(scale),
                                               ct.c_float(power))
    if err_code:
        raise generate_exception(err_code)

    return target

def crossmap_response_norm_undo(imgs, respGrads, respActs, target, channels, sizeX, scale, power):
    """
    Undo response normalization of kernel dimension sizeX on imgs and put result in target
    Images as  (CxHxW) Rows x (N) Columns in 'C' order 
    Target as  (KxPxQ) Rows x (N) Colums in 'C' order
    int channels, sizeX, float scale, power
    """

    err_code = _cudanet.crossmap_response_norm_undo(imgs.p_mat, respGrads.p_mat,
                                                    respActs.p_mat, target.p_mat,
                                                    ct.c_int(channels),
                                                    ct.c_int(sizeX),
                                                    ct.c_float(scale),
                                                    ct.c_float(power),
                                                    ct.c_float(0))
    if err_code:
        raise generate_exception(err_code)

    return target

def local_contrast_norm(imgs, meanDiffs, denoms, target, imgSizeX, channels, sizeX, scale, power):
    """
    Perform contrast normalization across channels of kernel dimension sizeX on
    imgs and put result in target
    Images as  (CxHxW) Rows x (N) Columns in 'C' order 
    Target as  (KxPxQ) Rows x (N) Colums in 'C' order
    int channels, sizeX, float scale, power
    """
    err_code = _cudanet.local_contrast_norm(imgs.p_mat,
                                            meanDiffs.p_mat,
                                            denoms.p_mat,
                                            target.p_mat,
                                            ct.c_int(imgSizeX),
                                            ct.c_int(channels),
                                            ct.c_int(sizeX),
                                            ct.c_float(scale),
                                            ct.c_float(power))
    if err_code:
        raise generate_exception(err_code)

    return target

def local_contrast_norm_undo(meanDiffs, denoms, respGrads, respActs, target, channels, sizeX, scale, power):
    """
    Undo contrast normalization of kernel dimension sizeX on imgs and put result in target
    Images as  (CxHxW) Rows x (N) Columns in 'C' order 
    Target as  (KxPxQ) Rows x (N) Colums in 'C' order
    int channels, sizeX, float scale, power
    """
    err_code = _cudanet.local_contrast_norm_undo(meanDiffs.p_mat, denoms.p_mat, 
                                                 respGrads.p_mat,
                                                 respActs.p_mat, target.p_mat,
                                                 ct.c_int(channels),
                                                 ct.c_int(sizeX),
                                                 ct.c_float(scale),
                                                 ct.c_float(power),
                                                 ct.c_float(0))
    if err_code:
        raise generate_exception(err_code)

    return target

def adadelta_update(grads, eGradSq, eDeltSq, deltX, rho, epsilon):
    """
    Does the in-place updates for adadelta_update
    Each matrix has the dimensions of the parameters to be updated, and rho and epsilon are scalars
    requires all matrix inputs to be predefined, grads will be unchanged, the other will be updated based on the following:
    eGradSq = eGradSq * rho + grads^2 * (1-rho)
    deltX = -sqrt((eDeltSq + epsilon)/(eGradSq + epsilon)) * grads
    eDeltSq = eDeltSq * rho + deltX^2 * (1-rho)
    """

    err_code = _cudanet.adadelta_update(grads.p_mat, eGradSq.p_mat, eDeltSq.p_mat,
                                        deltX.p_mat, ct.c_float(rho),
                                        ct.c_float(epsilon))
    if err_code:
        raise generate_exception(err_code)

    return deltX

def convolution(wts, imgs, target, imgSizeY, numModulesY, numModulesX,
                paddingStart, moduleStride, numImgColors, numGroups,
                doLocal=False):
    """
    Convolve wts with imgs and put result in target
	Weights as (CxRxS) Rows x (K) Columns in 'C' order
	Images as  (CxHxW) Rows x (N) Columns in 'C' order 
	Target as  (KxPxQ) Rows x (N) Colums in 'C' order
    	int imgSizeY, int numModulesX, int paddingStart, int moduleStride,
        int numImgColors, int numGroups
    """

    err_code = _cudanet.convolution(wts.p_mat, imgs.p_mat, target.p_mat,
                                    ct.c_int(imgSizeY),
                                    ct.c_int(numModulesY),
                                    ct.c_int(numModulesX),
                                    ct.c_int(paddingStart),
                                    ct.c_int(moduleStride),
                                    ct.c_int(numImgColors),
                                    ct.c_int(numGroups),
                                    ct.c_bool(doLocal));
    if err_code:
        raise generate_exception(err_code)

    return target


def deconvolve_errors(wts, errors, target, imgSizeY, imgSizeX, numModulesY,
                      paddingStart, moduleStride, numImgColors, numGroups,
                      doLocal=False):
    """
    Backprop errors and put result in target
    Weights as (CxRxS) Rows x (K) Columns in 'C' order
    Errors as  (KxPxQ) Rows x (N) Columns in 'C' order 
    Target as  (CxHxW) Rows x (N) Colums in 'C' order
        int imgSizeY, ing imgSizeX, int numModulesY, int paddingStart, 
        int moduleStride, int numImgColors, int numGroups
    """
    err_code = _cudanet.convolution_back_errors(wts.p_mat, errors.p_mat,
                                                target.p_mat, ct.c_int(imgSizeY),
                                                ct.c_int(imgSizeX),
                                                ct.c_int(numModulesY),
                                                ct.c_int(paddingStart),
                                                ct.c_int(moduleStride),
                                                ct.c_int(numImgColors),
                                                ct.c_int(numGroups),
                                                ct.c_float(0),
                                                ct.c_bool(doLocal));
    if err_code:
        raise generate_exception(err_code)

    return target

def deconvolve_wts(hidActs, imgs, target, imgSizeY, numModulesY, numModulesX,
                   filterSize, paddingStart, moduleStride, numImgColors,
                   numGroups, sumWidth, doLocal=False):
    """
    Backprop acts grad with img grad to compute wts grad and put result in target
    hidActs as  (CxHxW) Rows x (N) Columns in 'C' order 
    imgs as  (KxPxQ) Rows x (N) Colums in 'C' order
    Target as (CxRxS) Rows x (K) Columns in 'C' order
        int imgSizeY, ing numModulesY, int numModulesX, int filterSize,
        int paddingStart, int moduleStride, int numImgColors, int numGroups,
        int sumWidth
    """
    err_code = _cudanet.convolution_back_weights(hidActs.p_mat, imgs.p_mat,
                                                 target.p_mat, ct.c_int(imgSizeY),
                                                 ct.c_int(numModulesY),
                                                 ct.c_int(numModulesX),
                                                 ct.c_int(filterSize),
                                                 ct.c_int(paddingStart),
                                                 ct.c_int(moduleStride),
                                                 ct.c_int(numImgColors),
                                                 ct.c_int(numGroups),
                                                 ct.c_int(sumWidth),
                                                 ct.c_float(0),
                                                 ct.c_float(1),
                                                 ct.c_bool(doLocal));
    if err_code:
        raise generate_exception(err_code)

    return target

def xcov(X, Y, target = None, normX=1, normY=1, normAll=-1):
    """
    Find the xcov between X and Y and store in target:
    X is M1 x N
    Y is M2 x N

    target = 1/normAll * (X-mean(X)).T  dot (Y - mean(Y))

    by default, X and Y are mean normalized, meaning the column vector u_X (dim: M1x1)
    is subtracted from each column of X, and likewise for Y

    by default the entire quantity is normalized by N (for normAll = -1)

    If no target is given, it will be created automatically, but not
    initialized
    """

    if (normX != 0 and normX != 1):
        raise generate_exception(-6)
    if (normY != 0 and normY != 1):
        raise generate_exception(-6)    

    if (normAll == -1):
        normFactor = np.float32(_cudanet.get_nonleading_dimension(X.p_mat))
    else:
        normFactor = np.float32(normAll)

    if not target:
        m = _cudanet.get_leading_dimension(X.p_mat)
        n = _cudanet.get_leading_dimension(Y.p_mat)

        target = empty((m, n))

    err_code = _cudanet.xcov(X.p_mat, Y.p_mat, target.p_mat, ct.c_int(normX), ct.c_int(normY), ct.c_float(normFactor))
    if err_code:
        raise generate_exception(err_code)

    return target

# def cuda_set_device(dev_id):
#     """
#     Selects the CUDA device with the given ID.
#     """

#     err_code =  _cudanet.cuda_set_device(ct.c_int(dev_id))
#     if err_code:
#         raise generate_exception(err_code)

def split(mat, nsplit, axis):
    """
    Meant to provide functionality similar to vsplit and split in numpy
    Can split along either axis -- no default provided
    Not streamed optimally at the moment, everything happens sequentially
    each of the submats returned here have gpu buffers that are VIEWS of the 
    original, they are not copied, and therefore don't "own" their data
    """
    # Check validity of axis
    if (axis!=0 and axis!=1):
        raise generate_exception(-12)

    otheraxis = 1 if axis==0 else 0

    if (mat.shape[axis] % nsplit != 0):
        print("Dimension not divisible %d vs %d".format(mat.shape[axis],
                                                        nsplit))
        raise generate_exception(-12)

    subdim = mat.shape[axis]/nsplit

    split_list = []

    for i in range(nsplit):
        if (axis==0):
            split_list.append(mat.slice_view(subdim*i, subdim*(i+1), 0, mat.shape[1]))
        else:
            split_list.append(mat.slice_view(0, mat.shape[0], subdim*i, subdim*(i+1)))

    return split_list

def stack(cmats, axis, target = None):
    """
    Meant to provide functionality similar to vstack and hstack in numpy
    Can stack along either axis -- no default provided
    Not streamed optimally at the moment, everything happens sequentially
    """

    # Check validity of axis
    if (axis!=0 and axis!=1):
        raise generate_exception(-12)

    otheraxis = 1 if axis==0 else 0

    # Now check consistency of dimensions, samedims should have all values
    # be the same, since we are concatenating along the other dim
    samedims = [i.shape[otheraxis] for i in cmats]
    if (all([i==samedims[0] for i in samedims]) == False):
        print("Axes not the same along dimension %d".format(axis))
        raise generate_exception(-12)

    diffdims = [i.shape[axis] for i in cmats]

    if axis==0:
        totalshape = (sum(diffdims), samedims[0])
    else:
        totalshape = (samedims[0], sum(diffdims))

    if not target:
        target = empty(totalshape)
    else:
        if (target.shape != totalshape):
            print("Inconsistent shapes for existing target %s vs %s".format(
                  target.shape, totalshape))
            raise generate_exception(-1)

    idxs = np.insert(np.array(diffdims), 0, 0)
    idxs = np.cumsum(idxs)
    for (submat, startidx, endidx) in zip(cmats, idxs[:-1], idxs[1:]):
        if axis==0:
            target.set_row_slice(startidx, endidx, submat)
        else:
            target.set_col_slice(startidx, endidx, submat)

    return target

def multi_way_error(probs, labels, labellogprob, top1probs, topkprobs, topk):
    """
    probs is     numNodes x numExamples
    labels is    1 x numExamples
    labellogprob 1 x numExamples
    top1probs    1 x numExamples
    topkprobs    1 x numExamples

    """

    err_code = _cudanet.multi_ranked_error(probs.p_mat, labels.p_mat,
                                           labellogprob.p_mat, top1probs.p_mat,
                                           topkprobs.p_mat, ct.c_int(topk))
    if err_code:
        raise generate_exception(err_code)

def softmax(mat, target = None, axis=0):
    if not target:
        target = empty(mat.shape)
        
    err_code = _cudanet.softmax(mat.p_mat, target.p_mat, ct.c_int(axis))
    if err_code:
        raise generate_exception(err_code)

    return target

def softmax_grad(acts, actsGrad, target = None):
    if not target:
        target = empty(acts.shape)

    err_code = _cudanet.softmax_grad(acts.p_mat, actsGrad.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def crossent_cost(labels, outputs, target = None):
    if not target:
        n = _cudanet.get_nonleading_dimension(labels.p_mat)
        target = empty((1, n))

    err_code = _cudanet.crossent_cost(labels.p_mat, outputs.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def crossent_cost_grad(labels, outputs, target = None):
    if not target:
        target = empty(labels.shape)

    err_code = _cudanet.crossent_cost_grad(labels.p_mat, outputs.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target

def weight_norm_along_axis(weights, target = None, axis = 0, norm = 1.0):
    if not target:
        target = empty(weights.shape)

    err_code = _cudanet.weight_norm_along_axis(weights.p_mat, target.p_mat, ct.c_int(axis), ct.c_float(norm))
    if err_code:
        raise generate_exception(err_code)

    return target

def cudanet_init_random(seed = None):
    """
    Initialize and seed the random number generator.
    """
    if not seed:
        _cudanet.init_random_no_seed()
    else:
        _cudanet.init_random(ct.c_ulonglong(seed))

def cudanet_destroy_random():
    """
    Destroy the random number generator.
    """
    _cudanet.destroy_random()

def cublas_init():
    """
    Initialize Cublas.
    """
    _cudanet.cublas_init()
    CUDAMatrix.ones = CUDAMatrix(np.ones((MAX_ONES, 1), dtype=np.float32, order = 'C'))

# init = cublas_init

def cublas_shutdown():
    """
    Shut down Cublas.
    """
    CUDAMatrix.ones = 0
    _cudanet.cublas_shutdown()

# shutdown = cublas_shutdown

