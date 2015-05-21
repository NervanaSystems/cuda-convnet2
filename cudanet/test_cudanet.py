# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
import numpy as np
import cudanet
import nose
import math
from nose.plugins.attrib import attr

def make_links(ofmsize, nifm, fheight, fwidth, ifmheight, ifmwidth, stride):
    ifmsize = ifmheight * ifmwidth
    fsize = nifm * fheight * fwidth

    links = np.zeros((ofmsize, fsize), dtype='i32')
    # This variable tracks the top left corner of the receptive field.
    src = 0
    for dst in xrange(ofmsize):
        # Collect the column indices for the
        # entire receptive field.
        colinds = []
        for row in xrange(fheight):
            start = src + row * ifmwidth
            colinds += range(start, start + fwidth)
        fminds = colinds[:]
        for ifm in xrange(1, nifm):
            colinds += [x + ifm * ifmsize for x in fminds]

        if (src % ifmwidth + fwidth + stride) <= ifmwidth:
            # Slide the filter to the right by the stride value.
            src += stride
        else:
            # We hit the right edge of the input image.
            # Shift the filter down by one stride.
            src += stride * ifmwidth - src % ifmwidth
            assert src % ifmwidth == 0
        links[dst, :] = np.array(colinds, dtype='i32')
    # rlinks = links.raw()
    return links


def make_local_links(nifm, fheight, fwidth, ifmheight, ifmwidth, stride):
    ifmsize = ifmheight * ifmwidth
    ofmheight = (ifmheight - fheight) / stride + 1
    ofmwidth = (ifmwidth - fwidth) / stride + 1
    ofmsize = ofmheight * ofmwidth
    links = np.empty((ofmsize, fheight * fwidth), dtype='i32')
    # This variable tracks the top left corner of the receptive field.
    src = 0
    for dst in xrange(ofmsize):
        # Collect the column indices for the
        # entire receptive field.
        colinds = []
        for row in xrange(fheight):
            start = src + row * ifmwidth
            colinds += range(start, start + fwidth)
        fminds = colinds[:]

        if (src % ifmwidth + fwidth + stride) <= ifmwidth:
            # Slide the filter to the right by the stride value.
            src += stride
        else:
            # We hit the right edge of the input image.
            # Shift the filter down by one stride.
            src += stride * ifmwidth - src % ifmwidth
            assert src % ifmwidth == 0
        links[dst, :] = np.array(colinds, dtype='i32')
    return links


def hsse(outputs, targets):
    return 0.5 * np.sum((outputs - targets) ** 2)


def hsse_de(outputs, targets):
    return (outputs - targets)

def hsum(hmat):
    return hmat.sum()


def dsum(dmat):
    dresult = dmat.sum(axis=None)
    dresult.copy_to_host()
    return dresult.numpy_array[0][0]

def squish(data, nifm):
    assert data.shape[1] % nifm == 0
    return data.reshape((data.shape[0] * nifm, data.shape[1] / nifm))

def hstack_maps(obj, nfm):
    """
    Stack the feature maps horizontally.
    """
    assert obj.shape[0] % nfm == 0
    return np.hstack(np.vsplit(obj, nfm))

# Not part of the API - can be moved to a utility class.
def vstack_maps(obj, nfm):
    """
    Stack the feature maps vertically.
    """
    assert obj.shape[1] % nfm == 0
    return np.vstack(np.hsplit(obj, nfm))

class TestCudanet(object):

    def __init__(self):
        # this code gets called prior to each test
        self.be = cudanet
        cudanet.cublas_init()

    def __del__(self):
        cudanet.cublas_shutdown()


    @attr('pointer')
    def test_pointer(self):
        np.random.seed(0)
        m = 256
        n = 1
        cm1 = np.array(np.random.rand(n, m)*10, dtype=np.float32, order='C')
        gm1 = self.be.CUDAMatrix(cm1)

        gpuPointer = gm1.get_gpu_pointer()

        print(gpuPointer)

    @attr('reshape')
    def test_reshape(self):
        m = 10
        n = 1
        cm1 = np.array(np.random.rand(n, m)*10, dtype=np.float32, order='C')
        cm2 = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')

        gm1 = self.be.CUDAMatrix(cm1)

        a = gm1.reshape((2,5))
        a.set_row_slice(0,1, 0)

        cm2 = cm1.reshape((2,5))
        cm2[0] = 0
        cm2 = cm2.reshape(gm1.shape)
        gm1.copy_to_host()

        assert np.max(np.abs(gm1.numpy_array - cm2)) < 10**-2, "Error in CUDAMatrix.reshape exceeded threshold"

        print gm1.shape, gm1.numpy_array
        # gm2 = self.be.CUDAMatrix(cm2)

        # gm1.reshape((m, n))
        # gm2.assign(gm1)
        # gm1.reshape((n, m))

        # gm1.copy_to_host()
        # gm2.copy_to_host()

        # assert np.max(np.abs(gm1.numpy_array - gm2.numpy_array.T)) < 10**-2, "Error in CUDAMatrix.reshape exceeded threshold"

    @attr('dbm')
    def test_dbm(self):
        a = np.array(np.arange(20).reshape(10,2), dtype=np.float32, order='C')
        b = np.zeros_like(a)
        m1 = self.be.CUDAMatrix(a)
        m2 = self.be.CUDAMatrix(b)
        self.be.log(m1, target = m2)
        self.be.log(m1)

        m2.copy_to_host()
        print(m2.numpy_array)

    @attr('reshapeadd')
    def test_dbm2(self):
        a = np.array(np.arange(20).reshape(10,2), dtype=np.float32, order='C')
        b = np.zeros_like(a)
        c = np.array(np.ones(4).reshape(4,1), dtype=np.float32, order='C')
        m1 = self.be.CUDAMatrix(a)
        m2 = self.be.CUDAMatrix(b)
        m3 = self.be.CUDAMatrix(c)

        m1.reshape((4,5))
        m1.add(m3)
        m1.reshape((10,2))
        m1.copy_to_host()
        print(m1.numpy_array)

    @attr('mat_vec')
    def test_mat_vec(self):
        m = 5
        k = 3
        scale = 1.312
        a = np.array(np.random.randn(m, k)*10, dtype=np.float32, order='C')
        b = np.array(np.random.randn(m, 1)*10, dtype=np.float32, order='C')
        c = np.array(np.random.randn(1, k)*10, dtype=np.float32, order='C')

        o = np.zeros_like(a)
        d_a = self.be.CUDAMatrix(a.copy())
        d_b = self.be.CUDAMatrix(b.copy())
        d_c = self.be.CUDAMatrix(c.copy())

        d_t = self.be.CUDAMatrix(np.zeros_like(a))

        np.divide(a,b,out=o)
        d_a.divide(d_b, d_t)
        d_t.copy_to_host()
        errorv = np.linalg.norm(o - d_t.numpy_array)
        assert errorv < 10**-5, "Error in sum exceeded threshold"

        np.add(a,c,out=o)
        d_a.add(d_c, d_t)
        d_t.copy_to_host()
        errorv = np.linalg.norm(o - d_t.numpy_array)
        assert errorv < 10**-5, "Error in sum exceeded threshold"

    @attr('split')
    def test_split(self):
        np.set_printoptions(precision=4, linewidth=100, suppress=True)
        np.random.seed(0)
        a = np.array(np.random.rand(12,3)*10, dtype=np.float32, order='C')
        b = np.array(np.random.rand(3,12)*10, dtype=np.float32, order='C')

        da = self.be.CUDAMatrix(a)
        db = self.be.CUDAMatrix(b)

        da_subs = self.be.split(da, 4, axis=0)
        db_subs = self.be.split(db, 4, axis=1)

        for dd in da_subs:
            dd.copy_to_host()

        for dd in db_subs:
            dd.copy_to_host()

        dc = self.be.stack(da_subs, axis=0)
        dc.copy_to_host()
        dd = self.be.stack(db_subs, axis=1)
        dd.copy_to_host()

        errorv = np.linalg.norm(a - dc.numpy_array)
        errorh = np.linalg.norm(b - dd.numpy_array)

        assert errorh < 10**-2, "Error in CUDAMatrix.split exceeded threshold for hsplit"
        assert errorv < 10**-2, "Error in CUDAMatrix.split exceeded threshold for vsplit"


    @attr('topkcost')
    def test_topkcost(self):
        np.random.seed(0)
        m, n = 10, 6
        probs = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')
        labels = np.array([[6, 3, 6, 6, 4, 0]], dtype=np.float32, order='C')

        dprobs = self.be.CUDAMatrix(probs)
        dlabels = self.be.CUDAMatrix(labels)
        top1 = self.be.empty((1, n))
        topk = self.be.empty((1, n))
        labellog = self.be.empty((1, n))

        print probs
        print labels
        self.be.multi_way_error(dprobs, dlabels, labellog, top1, topk, 3)
        print top1.asarray()
        print topk.asarray()
        print labellog.asarray()



    @attr('stack')
    def test_stack(self):
        print("stack\n")
        rdims = [3,4,2,5]
        cdims = [2,5,4,3]

        Rdim = 3
        Cdim = 4

        vmats = []
        for i in range(len(rdims)):
            vmats.append(np.array(np.random.rand(rdims[i], Cdim)*10, dtype=np.float32, order='C'))

        hmats = []
        for i in range(len(cdims)):
            hmats.append(np.array(np.random.rand(Rdim, cdims[i])*10, dtype=np.float32, order='C'))


        h_all = np.hstack(hmats)
        v_all = np.vstack(vmats)

        d_vmats = []
        for i in range(len(rdims)):
            d_vmats.append(self.be.CUDAMatrix(vmats[i]))

        d_hmats = []
        for i in range(len(cdims)):
            d_hmats.append(self.be.CUDAMatrix(hmats[i]))

        d_h_all = self.be.stack(d_hmats, axis=1)
        d_v_all = self.be.stack(d_vmats, axis=0)

        d_h_all.copy_to_host()
        d_v_all.copy_to_host()

        errorh = np.linalg.norm(h_all - d_h_all.numpy_array)
        errorv = np.linalg.norm(v_all - d_v_all.numpy_array)

        assert errorh < 10**-2, "Error in CUDAMatrix.stack exceeded threshold for hstack"
        assert errorv < 10**-2, "Error in CUDAMatrix.stack exceeded threshold for vstack"

    def test_T_field(self):
        m = 256
        n = 128
        cm1 = np.array(np.random.rand(n, m)*10, dtype=np.float32, order='C')
        cm2 = np.array(np.random.rand(m, 1)*10, dtype=np.float32, order='C')
        gm1 = self.be.CUDAMatrix(cm1)
        gm2 = self.be.CUDAMatrix(cm2)

        # test dot
        gm = self.be.dot(gm2.T, gm1.T)
        c = np.dot(cm2.T, cm1.T)
        gm.copy_to_host()

        assert np.max(np.abs(gm.numpy_array - c)) < 10**-2, "Error in CUDAMatrix.dot with TransposedCUDAMatrix exceeded threshold"

    def test_T_field_dot1(self):
        print("\nT Field dot")

        m = 3
        n = 4
        k = 5
        cm1 = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')
        cm2 = np.array(np.random.rand(m, k)*10, dtype=np.float32, order='C')
        gm1 = self.be.CUDAMatrix(cm1)
        gm2 = self.be.CUDAMatrix(cm2)

        # test dot
        gm = self.be.dot(gm2.T, gm1)
        c = np.dot(cm2.T, cm1)
        gm.copy_to_host()
        error = np.linalg.norm(c - gm.numpy_array)
        print(error)
        print(gm.numpy_array)
        print(c)
        assert np.max(np.abs(gm.numpy_array - c)) < 10**-2, "Error in CUDAMatrix.dot with TransposedCUDAMatrix exceeded threshold"
        print("field dot ok")

    def test_assign_scalar(self):
        m = 256
        n = 128
        a = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')

        m1 = self.be.CUDAMatrix(a)

        m1.assign(np.pi)
        m1.copy_to_host()
        assert np.max(np.abs(m1.numpy_array - np.pi)) < 10**-4, "Error in CUDAMatrix.assign_scalar exceeded threshold"

    @attr('devcopy')
    def test_device_copy(self):
        print("\ndevice copy")
        m = 10
        n = 4
        a = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')
        m1 = self.be.empty(a.shape)
        m1.copy_to_host()

        m2 = self.be.empty(a.shape)
        m2.copy_to_host()

        m1.numpy_array[:] = np.arange(40).reshape(10,4)
        m2.numpy_array[:] = 2.

        m1.copy_to_device()
        m2.copy_to_device()

        m3 = m1.mult(m2)

        m3.copy_to_host()
        assert np.max(np.abs(np.arange(40).reshape(10,4)*2 - m3.numpy_array)) < 10**-4, "Error in device copy exceeded threshold"


    @attr('pybuf')
    def test_pybuf(self):
        print("\ntesting buffer")
        m = 10
        n = 4
        a = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')
        m1 = self.be.empty(a.shape)
        m1.copy_to_host()

        m1.numpy_array[:] = np.arange(40).reshape(10,4)
        m1.copy_to_device()

        m1buf = m1.get_gpu_pythonbuf()
        print "Returned after getting the buffer"
        dx = np.asarray(m1buf)
        # print dx.dtype
        # print m1buf[:2]

    @attr('log')
    def test_log(self):
        m = 256
        n = 128
        a = np.array(np.random.rand(m, n)*10+0.1, dtype=np.float32, order='C')
        b = np.array(np.random.rand(m, n)*10+0.1, dtype=np.float32, order='C')

        c = np.log(a)

        m1 = self.be.CUDAMatrix(a)
        m2 = self.be.CUDAMatrix(b)
        self.be.log(m1, target = m2)
        self.be.log(m1)

        m1.copy_to_host()
        m2.copy_to_host()

        assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in self.be.log exceeded threshold"
        assert np.max(np.abs(c - m2.numpy_array)) < 10**-4, "Error in self.be.log exceeded threshold"


    def test_assign(self):
        m = 256
        n = 128
        a = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')
        b = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')

        m1 = self.be.CUDAMatrix(a)
        m2 = self.be.CUDAMatrix(b)

        m1.assign(m2)
        m1.copy_to_host()

        assert np.max(np.abs(m1.numpy_array - m2.numpy_array)) < 10**-4, "Error in CUDAMatrix.assign exceeded threshold"

    @attr('row_slice')
    def test_get_row_slice(self):
        m = 256
        n = 128
        start = 11
        end = 54

        np.random.seed(0)
        ha = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')
        hslice = np.array(ha[start:end,:], order='C')

        da = self.be.CUDAMatrix(ha)
        dslice = self.be.CUDAMatrix(np.zeros(((end-start, n)),
                                             dtype=np.float32, order='C'))

        da.get_row_slice(start, end, target=dslice)
        dslice.copy_to_host()
        assert np.max(np.abs(hslice - dslice.numpy_array)) < 10**-4, "Error in CUDAMatrix.get_row_slice exceeded threshold"
        dslice = da.row_slice_view(start, end)
        dslice.copy_to_host()

        #pdb.set_trace()
        assert np.max(np.abs(hslice - dslice.numpy_array)) < 10**-4, "Error in CUDAMatrix.get_row_slice exceeded threshold"
        # Try dot after slice
        end = start + n
        ha = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')
        hslice = np.array(ha[start:end,:], order='C')

        da = self.be.CUDAMatrix(ha)
        dslice = da.row_slice_view(start, end)

        hprod = np.dot(hslice, hslice)
        dprod = self.be.CUDAMatrix(np.zeros(hprod.shape, dtype=np.float32))
        self.be.dot(dslice, dslice, target=dprod)
        dprod.copy_to_host()
        print hsum(hprod), dsum(dprod)
        assert np.max(np.abs(hprod - dprod.numpy_array)) < 10**-2, "Error in CUDAMatrix.get_row_slice exceeded threshold"

    @attr('col_slice')
    def test_get_col_slice(self):
        m = 4
        n = 8
        start = 2
        end = 4

        np.random.seed(0)
        ha = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')
        hslice = np.array(ha[:, start:end], order='C')

        da = self.be.CUDAMatrix(ha)
        dslice = self.be.CUDAMatrix(np.zeros(((end-start, n)),
                                             dtype=np.float32, order='C'))

        da.get_col_slice(start, end, target=dslice)
        dslice.copy_to_host()
        assert np.max(np.abs(hslice - dslice.numpy_array)) < 10**-4, "Error in CUDAMatrix.get_col_slice exceeded threshold"
        dslice = da.col_slice_view(start, end)
        dslice.copy_to_host()

        assert np.max(np.abs(hslice - dslice.numpy_array)) < 10**-4, "Error in CUDAMatrix.get_col_slice exceeded threshold"
        # Try dot after slice
        end = start + m
        ha = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')
        hslice = np.array(ha[:, start:end], order='C')

        da = self.be.CUDAMatrix(ha.copy())
        dslice = da.col_slice_view(start, end)
        dslice.copy_to_host()
        dslice.print_devmat()
        print dslice.numpy_array
        print hslice
        hprod = np.dot(hslice, hslice)
        dprod = self.be.CUDAMatrix(np.zeros(hprod.shape, dtype=np.float32))
        self.be.dot(dslice, dslice, target=dprod)
        dprod.copy_to_host()
        # print hsum(hprod), dsum(dprod)
        print dprod.numpy_array[0,:]
        print hprod[0,:]
        assert np.max(np.abs(hprod - dprod.numpy_array)) < 10**-2, "Error in CUDAMatrix.get_row_slice exceeded threshold"

    @attr('rslice')
    def test_set_row_slice(self):
        m = 256
        n = 128
        start = 11
        end = 54

        a = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')
        b = np.array(np.random.rand(end-start, n)*10, dtype=np.float32, order='C')

        c = a.copy()
        c[start:end,:] = b

        m1 = self.be.CUDAMatrix(a)
        m2 = self.be.CUDAMatrix(b)
        m1.set_row_slice(start, end, m2)
        m1.copy_to_host()

        assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.set_row_slice exceeded threshold"

        m1.set_row_slice(start, end, 4.2)
        m1.copy_to_host()

        c[start:end,:] = 4.2
        assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.set_row_slice exceeded threshold"

    @attr('cslice')
    def test_set_col_slice(self):
        m = 256
        n = 128
        start = 11
        end = 54

        a = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')
        b = np.array(np.random.rand(m, end-start)*10, dtype=np.float32, order='C')

        c = a.copy()
        c[:, start:end] = b

        m1 = self.be.CUDAMatrix(a)
        m2 = self.be.CUDAMatrix(b)
        m1.set_col_slice(start, end, m2)
        m1.copy_to_host()

        assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.set_col_slice exceeded threshold"

        m1.set_col_slice(start, end, 4.2)
        m1.copy_to_host()

        c[:, start:end] = 4.2
        assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.set_col_slice exceeded threshold"


    @attr('dot')
    def test_dot(self):
        print("\ndot")
        m = 4
        k = 3
        n = 2
        a = np.array(np.random.randn(m, k)*10, dtype=np.float32, order='C')
        b = np.array(np.random.randn(k, n)*10, dtype=np.float32, order='C')
        c = np.array(np.random.randn(m, n)*10, dtype=np.float32, order='C')

        alpha = 2.
        beta = 0.
        r = beta * c + alpha * np.dot(a, b)
        # print(r)
        m1 = self.be.CUDAMatrix(a)
        m2 = self.be.CUDAMatrix(b)
        m3 = self.be.CUDAMatrix(c)
        m3 = self.be.dot(m1, m2, target = m3, alpha = alpha, beta = beta)
        # m3.print_devmat()
        m3.copy_to_host()
        print(m3.numpy_array)
        assert np.max(np.abs(r - m3.numpy_array)) < 10**-2, "Error in CUDAMatrix.dot exceeded threshold"

    @attr('xcov')
    def test_xcov(self):
        print("\nxcov")
        np.random.seed(0)
        m = 4
        k = 3
        n = 10

        # Initialize with _numpy backend in mind (M points, K dimensions)
        a = np.array(np.random.randn(n, k)*10, dtype=np.float32, order='C')
        b = np.array(np.random.randn(n, m)*10, dtype=np.float32, order='C')
        c = np.zeros((k,m), dtype=np.float32)

        a0 = a-a.mean(0,keepdims=True)
        b0 = b-b.mean(0,keepdims=True)

        # This is the numpy version
        cv = a0.T.dot(b0)/np.float32(n)

        # in cudanet backend, the arrays will be k x m
        aT = a.T.copy()
        bT = b.T.copy()
        # # print(r)
        m1 = self.be.CUDAMatrix(aT)
        m2 = self.be.CUDAMatrix(bT)
        m3 = self.be.CUDAMatrix(c)
        m3 = self.be.xcov(m1, m2, target = m3)
        m3.copy_to_host()
        assert np.max(np.abs(cv - m3.numpy_array)) < 10**-2, "Error in CUDAMatrix.xcov exceeded threshold"

    @attr('xcovd2')
    def test_xcovd2(self):
        print("\nxcovd2")
        np.random.seed(0)
        m = 4
        k = 3
        n = 10

        # Initialize with _numpy backend in mind (M points, K dimensions)
        a = np.array(np.random.randn(n, k)*10, dtype=np.float32, order='C')
        b = np.array(np.random.randn(n, m)*10, dtype=np.float32, order='C')
        c = np.zeros((k,m), dtype=np.float32)

        a0 = a-a.mean(0,keepdims=True)
        b0 = b-b.mean(0,keepdims=True)

        # This is the numpy version
        cv = a0.T.dot(b0)/np.float32(n)

        # in cudanet backend, the arrays will be k x m
        aT = a.T.copy()
        bT = b.T.copy()
        # # print(r)
        m1 = self.be.CUDAMatrix(aT)
        m2 = self.be.CUDAMatrix(bT)
        m3 = self.be.CUDAMatrix(c)
        m3 = self.be.xcov(m1, m2, target = m3)
        m3.copy_to_host()
        # # print(cv)
        assert np.max(np.abs(cv - m3.numpy_array)) < 10**-2, "Error in CUDAMatrix.xcov exceeded threshold"



    @attr('hostcopy')
    def test_hostcopy(self):
        print("\nhostcopy")
        np.random.seed(0)
        m = 4
        k = 3
        n = 2
        a = np.array(np.random.randn(m, k)*10, dtype=np.float32, order='C')
        b = np.array(np.random.randn(n, k)*10, dtype=np.float32, order='C')


        alpha = 2.
        beta = 0.
        # r = beta * c + alpha * np.dot(a, b)
        # print(r    )
        m1 = self.be.CUDAMatrix(a)
        print("m1\n")
        m1.copy_to_host()
        print(m1.numpy_array)
        m2 = self.be.CUDAMatrix(b)
        m3 = self.be.dot(m1, m2.T, alpha = alpha, beta = beta)
        m3.copy_to_host()

        print("m1\n")
        m1.copy_to_host()
        print(m1.numpy_array)
        print(a)
        print("mslice\n")
        mslice = m1.get_row_slice(0,2)
        mslice.copy_to_host()
        print(mslice.numpy_array)


        m4 = self.be.dot(mslice, m2.T, alpha = alpha, beta = beta)
        print(m3.numpy_array)
        m4.copy_to_host()
        print("mslice * m2")
        print(m4.numpy_array)
        for batch in range(m):
            mslice = m1.get_row_slice(batch, batch+1)
            m4 = self.be.dot(mslice, m2.T, alpha = alpha, beta = beta)
            m4.copy_to_host()
            print(m4.numpy_array)


    def dsse(self, outputs, targets):
        diff = self.be.empty(outputs.shape)
        outputs.subtract(targets, target=diff)
        return (diff.mult(diff)).sum(axis=None).mult(0.5)


    def dsse_de(self, outputs, targets):
        diff = self.be.empty(outputs.shape)
        outputs.subtract(targets, target=diff)
        return diff


    @attr('mlp')
    def test_mlp(self):
        print("\nmlp")
        np.random.seed(0)
        N = 128
        H = 28
        W = 28
        eps = 0.000001
        nbatches = 2
        nepochs = 3
        nclasses = 10

        def f(nrows, ncols): return np.array(np.random.randn(nrows, ncols),
                                             dtype=np.float32)
        nin1 = H * W
        nout1 = 64
        nin2 = nout1
        nout2 = nclasses

        hdata = f(nbatches * N, H * W)
        hlabels = f(nbatches * N, nclasses)
        hweights1 = f(nout1, nin1)
        hweights2 = f(nout2, nin2)

        ddata = self.be.CUDAMatrix(hdata)
        dlabels = self.be.CUDAMatrix(hlabels)
        dweights1 = self.be.CUDAMatrix(hweights1.copy())
        dweights2 = self.be.CUDAMatrix(hweights2.copy())

        for epoch in range(nepochs):
            herror = 0.0
            derror = self.be.CUDAMatrix(np.zeros((1,1), dtype=np.float32))
            for batch in range(nbatches):
                # fprop
                hinputs = hdata[batch * N : (batch + 1) * N, :]
                htargets = hlabels[batch * N : (batch + 1) * N, :]
                houtputs1 = np.dot(hinputs, hweights1.T)
                houtputs2 = np.dot(houtputs1, hweights2.T)

                dinputs = ddata.get_row_slice(batch * N, (batch + 1) * N)
                dtargets = dlabels.get_row_slice(batch * N, (batch + 1) * N)
                doutputs1 = self.be.dot(dinputs, dweights1.T)
                doutputs2 = self.be.dot(doutputs1, dweights2.T)

                print('hwsum2\t', hsum(hweights2), '\tdwsum2\t',
                        dsum(dweights2))
                print('hsum2\t', hsum(houtputs2), '\tdsum2\t', dsum(doutputs2))
                assert (math.fabs(hsum(houtputs2) - dsum(doutputs2))) < 1.0

                # bprop
                herror += hsse(houtputs2, htargets)
                hgrads2 = hsse_de(houtputs2, htargets)
                hgrads1 = np.dot(hgrads2, hweights2)
                hweights2 -= eps * np.dot(hgrads2.T, houtputs1)
                hweights1 -= eps * np.dot(hgrads1.T, hinputs)

                derror.add(self.dsse(doutputs2, dtargets))
                dgrads2 = self.dsse_de(doutputs2, dtargets)
                dgrads1 = self.be.dot(dgrads2, dweights2)
                dweights2.subtract(self.be.dot(dgrads2.T, doutputs1,alpha=eps))
                dweights1.subtract(self.be.dot(dgrads1.T, dinputs,alpha=eps))
                # dweights2.subtract(self.be.dot(dgrads2.T, doutputs1).mult(eps))
                # dweights1.subtract(self.be.dot(dgrads1.T, dinputs).mult(eps))

            derror.copy_to_host()
            print('epoch', epoch, 'herror', herror, 'derror',
                    derror.numpy_array[0][0])

        print('mlp OK')

    @attr('maxpool')
    def test_maxpool(self):
        print("\nmaxpool")
        np.random.seed(0)
        N = 1        # number of images
        H = 6         # image height
        W = 6         # image width
        R = 3          # filter height
        S = R          # filter width
        C = 16          # image and filter depths
        P = H-R+1      # output feature map height (mode='valid')
        Q = W-S+1      # output feature map width
        stride = 1
        ofmsize = P*Q
        psize = R*S
        def f(args): return np.random.randn(*args)
        # def f(args): return np.arange(np.prod(args)).reshape(args)
        data = np.array(f((N,C,H,W))*10, dtype=np.single, order='C')
        result = np.array(np.empty((N,C,P,Q)), dtype=np.single, order='C')
        grads = np.array(np.abs(f((N,C,P,Q))), dtype=np.single, order='C')
        backgrads = np.array(np.empty((N,C,H,W)), dtype=np.single, order='C')

        links = np.zeros((ofmsize, psize), dtype='i32')
        src = 0 # This variable tracks the top left corner
                # of the receptive field.
        for dst in range(ofmsize):
            colinds = []
            # Collect the column indices for the
            # entire receptive field.
            for row in range(R):
                start = src + row * W
                colinds += range(start, start + S) 
            src += stride
            if ((src+S-1) % W) == 0:
                # Shift the pooling window down by 1 receptive field.
                src += (stride) * (S-1)
            links[dst] = colinds

        Tkz = self.be.CUDAMatrix(result.reshape((N,C*P*Q)).transpose().copy())
        Ikz = self.be.CUDAMatrix(data.reshape((N,C*H*W)).transpose().copy())
        Gkz = self.be.CUDAMatrix(grads.reshape((N,C*P*Q)).transpose().copy())
        Bkz = self.be.CUDAMatrix(backgrads.reshape((N,C*H*W)).transpose().copy())

        self.be.max_pool(Ikz, Tkz, C, sizeX=R, paddingStart=0, moduleStride=1, numModulesX=Q)
        Tkz.copy_to_host()
        a = Tkz.numpy_array.transpose().reshape(N,C,P,Q).copy()

        self.be.max_pool_undo(Ikz, Gkz, Tkz, Bkz, sizeX=R, paddingStart=0, moduleStride=1, numModulesX=Q)
        Bkz.copy_to_host()
        b = Bkz.numpy_array.transpose().reshape(N,C,H,W).copy()
        np.set_printoptions(precision=3)
        i=1
        print("inputs\n", data[0,i,:,:])
        print("errors\n", grads[0,i,:,:])
        print("maxActs\n", a[0,i,:,:])
        print("maxGrads\n", b[0,i,:,:])

        # error = np.linalg.norm(C2 - Tkz.numpy_array)
        # relerror = error/np.mean(np.abs(C2))
        # print("convolution error %f " % relerror)

        # assert abs(relerror) < 1e-3, "Error in self.be.convolution exceeded threshold"

    @attr('adadelta')
    def test_adadelta(self):
        print("\nadadelta")
        np.random.seed(0)
        # Just checking mechanics

        Ni = 4          # Num iterations

        Nf = 2
        R = 3         # filter height
        S = R          # filter width
        C = 2          # image and filter depths

        rho = 0.9
        epsilon = .01

        def f(args): return np.random.randn(*args)
        # def f(args): return np.arange(np.prod(args)).reshape(args)
        grads = np.array(f((Ni,Nf,C*R*S)), dtype=np.single, order='C')

        eGradSq = np.array(np.zeros((Nf,C*R*S)), dtype=np.single, order='C')
        eDeltSq = np.array(np.zeros((Nf,C*R*S)), dtype=np.single, order='C')
        deltX = np.array(np.zeros((Nf,C*R*S)), dtype=np.single, order='C')

        for i in range(Ni):
            eGradSq = rho * eGradSq + (1-rho) * grads[i,:,:] * grads[i,:,:]
            deltX = - np.sqrt((eDeltSq + epsilon)/(eGradSq + epsilon)) * grads[i,:,:]
            eDeltSq = rho * eDeltSq + (1-rho) * deltX * deltX
            # npAdaDeltaUpdate(grads[i,:,:], eGradSq, eDeltSq, deltX, rho, epsilon)

        d_eGradSq = self.be.CUDAMatrix(np.zeros_like(deltX))
        d_eDeltSq = self.be.CUDAMatrix(np.zeros_like(deltX))
        d_deltX =   self.be.CUDAMatrix(np.zeros_like(deltX))

        for i in range(Ni):
            d_grads = self.be.CUDAMatrix(grads[i,:,:])
            self.be.adadelta_update(d_grads, d_eGradSq, d_eDeltSq, d_deltX, rho, epsilon)

        d_deltX.copy_to_host()

        error = np.linalg.norm(deltX - d_deltX.numpy_array)
        relerror = error/np.mean(np.abs(deltX))
        print("adadelta error %f " % relerror)

        assert abs(relerror) < 1e-5, "Error in self.be.adadelta exceeded threshold"


    @attr('convslice')
    def test_convolution_slice(self):
        print("\nconvolution filter activations")
        np.random.seed(0)

        N = 128        # number of images
        H = 10         # image height
        W = 10         # image width
        K = 16         # number of filters
        R = 3          # filter height
        S = R          # filter width
        C = 3          # image and filter depths

        P = H-R+1      # output feature map height (mode='valid')
        Q = W-S+1      # output feature map width

        def f(args): return np.random.randn(*args)

        data = np.array(f((N,C,H,W)), dtype=np.single, order='C')
        filters = np.array(f((K,C,R,S)), dtype=np.single, order='C')
        result = np.array(np.empty((N,K,P,Q)), dtype=np.single, order='C')

        # First calculate using tensordot
        ksub = 3
        result.fill(0)
        for nn,a,b in np.ndindex(N,R,S):
            result[nn, :ksub, :, :] += np.tensordot(filters[:ksub,:,a,b], data[nn,:,a:a+P,b:b+Q], 1)

        # Now reshape everything into the tensor format that cudaconvnet expects
        # (channels x spatial1 x spatial2) x minibatch size
        C2 = result.transpose(1,2,3,0).reshape((K*P*Q,N)).copy()

        Tkz = self.be.CUDAMatrix(np.zeros_like(C2))
        Wkz = self.be.CUDAMatrix(filters.reshape((K,C*R*S)).transpose().copy())
        Ikz = self.be.CUDAMatrix(data.reshape((N,C*H*W)).transpose().copy())
        # Wkz = self.be.CUDAMatrix(filters.reshape((K,C*R*S))).T
        # Ikz = self.be.CUDAMatrix(data.reshape((N,C*H*W))).T



        C2 = result[:,:ksub,:,:].transpose(1,2,3,0).reshape((ksub*P*Q,N)).copy()
        self.be.convolution(Wkz, Ikz, Tkz, H, P, Q, 0, 1, C, 1)
        Tkz.copy_to_host()
        error = np.linalg.norm(C2 - Tkz.numpy_array.reshape(K,P,Q,N)[:ksub,:,:,:].reshape(ksub*P*Q,N))
        relerror = error/np.mean(np.abs(C2))
        print("convolution error %f " % relerror)
        assert abs(relerror) < 1e-3, "Error in self.be.convolution exceeded threshold"


        print("\nconvolution image activations")
        ofmsize = P * Q
        ofmstarts = np.array(range(0, (ofmsize * K), ofmsize))
        ofmlocs = np.zeros((ofmsize, K), dtype='i32')
        for dst in xrange(ofmsize):
            ofmlocs[dst, :] = ofmstarts + dst
        senderror = result.copy()
        berror = np.zeros_like(data)
        links = make_links(ofmsize, C, R, S, H, W, 1)
        senderror.shape = (N, K*P*Q)
        berror.shape = (N, C*H*W)
        filters.shape = (K, C*R*S)
        for dst in xrange(ofmsize):
            rflinks = links[dst]
            berror[:, links[dst]] +=np.dot(senderror.take(ofmlocs[dst], axis = 1), filters)
        berror.shape = (N,C,H,W)


        Ekz = self.be.CUDAMatrix(senderror.transpose().copy()) # Same as senderror
        Tkz = self.be.empty((C*H*W,N))
        self.be.deconvolve_errors(Wkz, Ekz, Tkz, H, W, P, 0, 1, C, 1)

        B2 = berror.transpose(1,2,3,0).reshape((C*H*W,N)).copy()
        Tkz.copy_to_host()
        error = np.linalg.norm(B2 - Tkz.numpy_array)
        relerror/np.mean(np.abs(B2))

        print("convolution error %f " % relerror)
        assert abs(relerror) < 1e-3, "Error in self.be.convolution exceeded threshold"


        print("\nconvolution weight updates")
        updates = np.zeros_like(filters)
        data.shape = (N, C*H*W)
        Ikz = self.be.CUDAMatrix(data.transpose().copy())
        Ukz = self.be.empty(Wkz.shape)
        for dst in xrange(ofmsize):
            # Accumulate the weight updates, going over all
            # corresponding cells in the output feature maps.
            rflinks = links[dst]
            eslice = senderror.take(ofmlocs[dst], axis=1)
            updates += np.dot(eslice.T, data.take(rflinks, axis=1))

        #Check filtersize -- one dimension or product
        self.be.deconvolve_wts(Ekz, Ikz, Ukz, H, P, Q, R, 0, 1, C, 1, sumWidth=P)

        updates2 = updates.transpose().copy()
        Ukz.copy_to_host()
        error = np.linalg.norm(updates2 - Ukz.numpy_array)
        # print(senderror.T[0,:10])
        # print(Ekz.numpy_array[0,:10])

        # print(data.T[0,:10])
        # print(Ikz.numpy_array[0,:10])
        relerror = error/np.mean(np.abs(updates2))
        print("convolution error %f " % relerror)
        assert abs(relerror) < 1e-3, "Error in self.be.convolution exceeded threshold"

    @attr('conv')
    def test_convolution(self):
        print("\nconvolution filter activations")
        np.random.seed(0)

        N = 128        # number of images
        H = 32         # image height
        W = 32         # image width
        K = 16         # number of filters
        R = 5          # filter height
        S = R          # filter width
        C = 3          # image and filter depths

        P = H-R+1      # output feature map height (mode='valid')
        Q = W-S+1      # output feature map width

        def f(args): return np.random.randn(*args)
        def f1(args): return np.random.uniform(-1.0, 1.0, args)
        def f2(args): return np.random.uniform(-.10, .10, args)

        # data = np.array(f1((N,C,H,W)), dtype=np.single, order='C')
        # filters = np.array(f2((K,C,R,S)), dtype=np.single, order='C')
        result = np.array(np.empty((N,K,P,Q)), dtype=np.single, order='C')
        # cpuE = np.array(f1((N,K,P,Q)), dtype=np.single, order='C')
        savefile = '/home/users/alex/Code/flexgpu/saved_conv_stats.pkl'
        import cPickle
        with open(savefile, 'r') as fo:
            idict = cPickle.load(fo)
        data = idict['I'][:-1].reshape((C,H,W,N)).transpose(3,0,1,2).copy()
        filters = idict['F'].reshape((C,R,S,K)).transpose(3,0,1,2).copy()
        cpuE = idict['E'].reshape((K,P,Q,N)).transpose(3,0,1,2).copy()
        # First calculate using tensordot
        result.fill(0)
        for nn,a,b in np.ndindex(N,R,S):
            result[nn] += np.tensordot(filters[:,:,a,b], data[nn,:,a:a+P,b:b+Q], 1)

        # Now reshape everything into the tensor format that cudaconvnet expects
        # (channels x spatial1 x spatial2) x minibatch size
        C2 = result.transpose(1,2,3,0).reshape((K*P*Q,N)).copy()

        Tkz = self.be.CUDAMatrix(np.zeros_like(C2))
        Wkz = self.be.CUDAMatrix(filters.reshape((K,C*R*S)).transpose().copy())
        Ikz = self.be.CUDAMatrix(data.reshape((N,C*H*W)).transpose().copy())
        # Wkz = self.be.CUDAMatrix(filters.reshape((K,C*R*S))).T
        # Ikz = self.be.CUDAMatrix(data.reshape((N,C*H*W))).T

        self.be.convolution(Wkz, Ikz, Tkz, H, P, Q, 0, 1, C, 1)
        Tkz.copy_to_host()
        error = np.linalg.norm(C2 - Tkz.numpy_array)
        print Tkz.numpy_array
        print '\n'
        print idict['fpropgpu'].reshape((K,P,Q,N)).transpose(3,0,1,2).reshape((K*P*Q,N))
        relerror = error/np.mean(np.abs(C2))
        print("convolution error %f " % relerror)

        assert abs(relerror) < 1e-3, "Error in self.be.convolution exceeded threshold"


        print("\nconvolution image activations")
        ofmsize = P * Q
        ofmstarts = np.array(range(0, (ofmsize * K), ofmsize))
        ofmlocs = np.zeros((ofmsize, K), dtype='i32')
        for dst in xrange(ofmsize):
            ofmlocs[dst, :] = ofmstarts + dst
        senderror = cpuE
        berror = np.zeros_like(data)
        links = make_links(ofmsize, C, R, S, H, W, 1)
        senderror.shape = (N, K*P*Q)
        berror.shape = (N, C*H*W)
        filters.shape = (K, C*R*S)
        for dst in xrange(ofmsize):
            rflinks = links[dst]
            berror[:, links[dst]] +=np.dot(senderror.take(ofmlocs[dst], axis = 1), filters)
        berror.shape = (N,C,H,W)


        Ekz = self.be.CUDAMatrix(senderror.transpose().copy()) # Same as senderror
        Tkz = self.be.empty((C*H*W,N))
        self.be.deconvolve_errors(Wkz, Ekz, Tkz, H, W, P, 0, 1, C, 1)

        B2 = berror.transpose(1,2,3,0).reshape((C*H*W,N)).copy()
        Tkz.copy_to_host()
        error = np.linalg.norm(B2 - Tkz.numpy_array)
        relerror/np.mean(np.abs(B2))

        print("convolution error %f " % relerror)
        assert abs(relerror) < 1e-3, "Error in self.be.convolution exceeded threshold"


        print("\nconvolution weight updates")
        updates = np.zeros_like(filters)
        data.shape = (N, C*H*W)
        Ikz = self.be.CUDAMatrix(data.transpose().copy())
        Ukz = self.be.empty(Wkz.shape)
        for dst in xrange(ofmsize):
            # Accumulate the weight updates, going over all
            # corresponding cells in the output feature maps.
            rflinks = links[dst]
            eslice = senderror.take(ofmlocs[dst], axis=1)
            updates += np.dot(eslice.T, data.take(rflinks, axis=1))

        #Check filtersize -- one dimension or product
        self.be.deconvolve_wts(Ekz, Ikz, Ukz, H, P, Q, R, 0, 1, C, 1, sumWidth=5)

        updates2 = updates.transpose().copy()
        Ukz.copy_to_host()
        error = np.linalg.norm(updates2 - Ukz.numpy_array)
        # print(senderror.T[0,:10])
        # print(Ekz.numpy_array[0,:10])

        # print(data.T[0,:10])
        # print(Ikz.numpy_array[0,:10])
        relerror = error/np.mean(np.abs(updates2))
        print("convolution error %f " % relerror)
        assert abs(relerror) < 1e-3, "Error in self.be.convolution exceeded threshold"


    @attr('add')
    def test_add(self):
        print('\nadd:')
        np.random.seed(0)
        m = 5
        k = 3
        scale = 1.312
        a = np.array(np.random.randn(m, k)*10, dtype=np.float32, order='C')
        b = np.array(np.random.randn(m, 1)*10, dtype=np.float32, order='C')
        c = np.array(np.random.randn(1, k)*10, dtype=np.float32, order='C')

        o = np.zeros_like(a)
        d_a = self.be.CUDAMatrix(a.copy())
        d_b = self.be.CUDAMatrix(b.copy())
        d_c = self.be.CUDAMatrix(c.copy())

        d_t = self.be.CUDAMatrix(np.zeros_like(a))

        np.add(a,b,out=o)
        d_a.add(d_b, d_t)
        d_t.copy_to_host()
        errorv = np.linalg.norm(o - d_t.numpy_array)
        assert errorv < 10**-5, "Error in sum exceeded threshold"

        np.add(a,c,out=o)
        d_a.add(d_c, d_t)
        d_t.copy_to_host()
        errorv = np.linalg.norm(o - d_t.numpy_array)
        assert errorv < 10**-5, "Error in sum exceeded threshold"

        np.add(a,c,out=a)
        d_a.add(d_c)
        d_a.copy_to_host()
        errorv = np.linalg.norm(a - d_a.numpy_array)
        assert errorv < 10**-5, "Error in sum exceeded threshold"

        np.add(a, c*scale, out=a)
        d_a.add_vector(d_c, scale)
        d_a.copy_to_host()
        errorv = np.linalg.norm(a - d_a.numpy_array)
        assert errorv < 10**-5, "Error in sum exceeded threshold"

    @attr('sum')
    def test_sum(self):
        print('\nsum:')
        np.random.seed(0)
        m = 4
        k = 3
        n = 2
        a = np.array(np.random.randn(m, k)*10, dtype=np.float32, order='C')

        hval = a.sum()
        m1 = self.be.CUDAMatrix(a)
        m2 = m1.sum(axis=0)
        m3 = m2.sum(axis=1)
        m4 = m1.sum(axis=None)
        m2.copy_to_host()
        m3.copy_to_host()
        m4.copy_to_host()
        # print(m2.numpy_array, a.sum(axis=0))
        # print(m3.numpy_array, a.sum(axis=1)    )
        dval = m3.numpy_array[0][0]
        dval2 = m4.numpy_array[0][0]
        print('hval', hval, 'dval', dval, 'dval2', dval2)

        assert np.abs(hval - dval) < 10**-2, "Error in sum exceeded threshold"
        assert np.abs(hval - dval2) < 10**-2, "Error in sumGlobal exceeded threshold"
        print('sum OK')

    @attr('sumsq')
    def test_sumsq(self):
        print('\nsumsq:')
        np.random.seed(0)
        m = 4
        k = 3
        n = 2
        a = np.array(np.random.randn(m, k)*10, dtype=np.float32, order='C')
        asq = a*a
        m1 = self.be.CUDAMatrix(a)

        m2 = m1.sumsq(axis=0)
        h2 = asq.sum(axis=0)

        m3 = m1.sumsq(axis=1)
        h3 = asq.sum(axis=1, keepdims=True)

        m4 = m1.sumsq(axis=None)
        hval = asq.sum()

        m2.copy_to_host()
        m3.copy_to_host()
        m4.copy_to_host()

        dval = m4.numpy_array[0][0]

        assert np.abs(hval - dval) < 10**-2, "Error in sumsq exceeded threshold"
        assert np.max(np.abs(h2 - m2.numpy_array)) < 10**-3, "Error in sumsq exceeded threshold"
        assert np.max(np.abs(h3 - m3.numpy_array)) < 10**-3, "Error in sumsq exceeded threshold"
        print('sumsq OK')

    def test_min(self):
        print('\nmin:')
        np.random.seed(0)
        m = 4
        k = 3
        a = np.array(np.random.randn(m, k)*10, dtype=np.float32, order='C')

        hval = a.min()
        m1 = self.be.CUDAMatrix(a)
        m2 = m1.min(axis=0)
        m3 = m2.min(axis=1)
        m4 = m1.min(axis=None)
        m3.copy_to_host()
        m4.copy_to_host()
        dval = m3.numpy_array[0][0]
        dval2 = m4.numpy_array[0][0]
        print('hval', hval, 'dval', dval, 'dval2', dval2)

        assert np.abs(hval - dval) < 10**-2, "Error in min exceeded threshold"
        assert np.abs(hval - dval2) < 10**-2, "Error in minGlobal exceeded threshold"

        print('min OK')

    @attr('mean')
    def test_mean(self):
        print('\nmean:')
        np.random.seed(0)

        m = 5
        n = 3
        a = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')
        t1 = np.array(np.random.rand(1, n)*10, dtype=np.float32, order='C')
        t2 = np.array(np.random.rand(m, 1)*10, dtype=np.float32, order='C')

        c1 = np.atleast_2d(a.mean(0))
        c2 = np.atleast_2d(a.mean(1)).T

        m = self.be.CUDAMatrix(a)
        mt1 = self.be.CUDAMatrix(t1)
        mt2 = self.be.CUDAMatrix(t2)
        m.mean(axis = 0, target = mt1)
        mt1r = m.mean(axis = 0)

        m.mean(axis = 1, target = mt2)
        mt2r = m.mean(axis = 1)

        mt1.copy_to_host()
        mt1r.copy_to_host()
        mt2.copy_to_host()
        mt2r.copy_to_host()
        assert np.max(np.abs(c1 - mt1.numpy_array)) < 10**-3, "Error in CUDAMatrix.mean exceeded threshold"
        assert np.max(np.abs(c1 - mt1r.numpy_array)) < 10**-3, "Error in CUDAMatrix.mean exceeded threshold"
        assert np.max(np.abs(c2 - mt2.numpy_array)) < 10**-3, "Error in CUDAMatrix.mean exceeded threshold"
        assert np.max(np.abs(c2 - mt2r.numpy_array)) < 10**-3, "Error in CUDAMatrix.mean exceeded threshold"

        print('mean OK')
        print('\nvar:')
        c1 = np.atleast_2d(a.var(0))
        mm1 = mt1.copy()
        m.var(axis=0, mean=mm1, target=mt1)

        c2 = np.atleast_2d(a.var(1)).T
        mm2 = mt2.copy()
        m.var(axis=1, mean=mm2, target=mt2)

        assert np.max(np.abs(c1 - mt1.asarray())) < 10**-3, "Error in CUDAMatrix.mean exceeded threshold"
        assert np.max(np.abs(c2 - mt2.asarray())) < 10**-3, "Error in CUDAMatrix.mean exceeded threshold"
        print('var OK')

    @attr('tot_norm')
    def test_tot_norm(self):
        print('\ntot norm:')
        np.random.seed(0)

        m = 5
        n = 3
        a = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')
        m_a0 = a.sum(axis=0, keepdims=True)
        d = self.be.CUDAMatrix(a.copy())
        m_d0 = d.sum(axis=0)
        d.divide(m_d0, target=d)
        np.divide(a, m_a0, out=a)

        assert np.max(np.abs(a - d.asarray())) < 10**-3, "Error in CUDAMatrix.div broadcast exceeded threshold"

    @attr('mean_norm')
    def test_mean_norm(self):
        print('\nmean norm:')
        np.random.seed(0)

        m = 5
        n = 3
        a = np.array(np.random.rand(m, n)*10, dtype=np.float32, order='C')

        m_a0 = a - a.mean(0,keepdims=True)
        m_a1 = a - a.mean(1,keepdims=True)
        m_aa = a - a.mean()

        d = self.be.CUDAMatrix(a)
        m_d0 = d.mean_norm(axis=0)
        m_d1 = d.mean_norm(axis=1)
        m_da = d.mean_norm(axis=-1)

        d.copy_to_host()
        m_d0.copy_to_host()
        m_d1.copy_to_host()
        m_da.copy_to_host()

        assert np.max(np.abs(m_a0 - m_d0.numpy_array)) < 10**-3, "Error in CUDAMatrix.mean exceeded threshold"
        assert np.max(np.abs(m_a1 - m_d1.numpy_array)) < 10**-3, "Error in CUDAMatrix.mean exceeded threshold"
        assert np.max(np.abs(m_aa - m_da.numpy_array)) < 10**-3, "Error in CUDAMatrix.mean exceeded threshold"

        print('mean norm OK')



    @attr('cmrnorm')
    def test_crossmap_response_norm(self):
        print('\ncrossmap response norm:')
        np.random.seed(0)
        N = 128       # number of images
        H = 4         # image height
        W = 4         # image width
        K = 3         # Kernel span
        
        C = 16        # image and filter depths
        r_scale = .0011
        r_power = 0.75
        epsilon = np.finfo(np.float32).eps
        def f(args): return np.fabs(np.random.randn(*args)*100)
        def g(args): return np.fabs(np.random.randn(*args)*10)

        inputs = np.array(f((C*H*W,N)), dtype=np.single, order='C')
        outputs = np.array(np.empty((C*H*W,N)), dtype=np.single, order='C')
        berror = np.array(np.empty((C*H*W,N)), dtype=np.single, order='C')
        error = np.array(g((C*H*W,N)), dtype=np.single, order='C')
        denoms = np.array(np.empty((C*H*W,N)), dtype=np.single, order='C')
        temp = np.array(np.empty((C*H*W,N)), dtype=np.single, order='C')

        rinputs = inputs.reshape((C,H,W,N))
        routputs = outputs.reshape((C,H,W,N))
        rerror = error.reshape((C,H,W,N))
        rberror = berror.reshape((C,H,W,N))
        rtemp = temp.reshape((C,H,W,N))
        rdenom = denoms.reshape((C,H,W,N))
        for i in xrange(C):
            knlidx = range(max(i-K/2,0), min(i-K/2+K,C))
            X = rinputs.take(knlidx, axis=0)
            np.square(X).sum(axis=0, out=routputs[i,:,:,:])
        
        # routputs[i,:,:,:] = np.power(1+r_scale*temp, -r_power)
        np.power(1+r_scale*outputs, -r_power, out=outputs)
        np.multiply(inputs, outputs, outputs)
        self.eps = 1e-4
        # houtputs = outputs.reshape((N,C*H*W)).transpose()
        d_inputs = self.be.CUDAMatrix(inputs.copy())
        d_inputs2 = self.be.CUDAMatrix(inputs.copy()+self.eps)
        d_outputs = self.be.CUDAMatrix(np.array(np.empty((C*H*W,N)), dtype=np.single, order='C'))
        d_outputs2 = self.be.CUDAMatrix(np.array(np.empty((C*H*W,N)), dtype=np.single, order='C'))
        d_error = self.be.CUDAMatrix(error.copy())
        d_berror = self.be.CUDAMatrix(np.array(np.empty((C*H*W,N)), dtype=np.single, order='C'))

        self.be.crossmap_response_norm(d_inputs, d_outputs, C, K, r_scale, r_power)
        self.be.crossmap_response_norm(d_inputs2, d_outputs2, C, K, r_scale, r_power)

        d_outputs.copy_to_host()
        assert np.max(np.abs(outputs - d_outputs.numpy_array)) < 10**-5, "Error in CUDAMatrix.crossmap_response_norm exceeded threshold"

        print('\ncrossmap response norm undo:')
        outputs = d_outputs.numpy_array.copy()
        routputs = outputs.reshape((C,H,W,N))

        for i in xrange(C):
            knlidx = range(max(i-K/2,0), min(i-K/2+K,C))
            X = rinputs.take(knlidx, axis=0)
            np.square(X).sum(axis=0, out=rdenom[i,:,:,:])
            knlidx2 = range(max(i-K + K/2+1,0), min(i+K/2+1,C))
            grad = rerror.take(knlidx2, axis=0)
            inp = rinputs.take(knlidx2, axis=0)
            act = routputs.take(knlidx2, axis=0)
            # (grad*inp).sum(axis=0, out=rtemp[i,:,:,:])
            (grad*act*np.power(act/inp, 1.0/r_power)).sum(axis=0, out=rtemp[i,:,:,:])

        np.multiply(temp, -2.0*r_power*r_scale, out=temp)
        np.multiply(temp, inputs, out=temp)

        # This is calculating 1+alpha*sumsq x_i
        np.power(1+r_scale*denoms, r_power, out=berror)

        np.divide(error, berror, out=berror)
        np.add(berror, temp, out=berror)

        self.be.crossmap_response_norm_undo(d_inputs, d_error, d_outputs,
                                            d_berror, C, K, r_scale, r_power)
        d_berror.copy_to_host()

        # print d_berror.numpy_array[:20,1]
        # print berror[:20,1]
        # print np.max(np.abs(berror - d_berror.numpy_array))
        assert np.max(np.abs(berror - d_berror.numpy_array)) < 10**-5, "Error in CUDAMatrix.crossmap_response_norm exceeded threshold"
        print('crossmap response norm OK')


    @attr('lcnnorm')
    def test_local_contrast_norm(self):
        print('\nlocal contrast norm:')
        np.random.seed(0)
        N = 128       # number of images
        H = 4         # image height
        W = 4         # image width
        K = 2         # Kernel span
        
        C = 16        # image and filter depths
        r_scale = 1
        r_power = 1
        epsilon = np.finfo(np.float32).eps
        def f(args): return np.fabs(np.random.randn(*args)*100)
        def g(args): return np.fabs(np.random.randn(*args)*10)

        inputs = np.array(f((C*H*W,N)), dtype=np.single, order='C')
        outputs = np.array(np.empty((C*H*W,N)), dtype=np.single, order='C')
        berror = np.array(np.empty((C*H*W,N)), dtype=np.single, order='C')
        error = np.array(g((C*H*W,N)), dtype=np.single, order='C')
        denoms = np.array(np.empty((C*H*W,N)), dtype=np.single, order='C')
        meandiffs = np.array(np.empty((C*H*W,N)), dtype=np.single, order='C')

        rinputs = inputs.reshape((C,H,W,N))
        routputs = outputs.reshape((C,H,W,N))
        rerror = error.reshape((C,H,W,N))
        rberror = berror.reshape((C,H,W,N))
        rdenom = denoms.reshape((C,H,W,N))
        rmeandiff = meandiffs.reshape((C,H,W,N))

        for y in xrange(H):
            yidx = range(max(y-K/2,0), min(y-K/2+K,H))
            hh = len(yidx)
            for x in xrange(W):
                xidx = range(max(x-K/2,0), min(x-K/2+K,W))
                ww = len(xidx)
                patch = rinputs.take(xidx, axis=1).take(yidx, axis=2).reshape(C, hh, ww, N)
                rmeandiff[:,x,y,:] = rinputs[:,x,y,:] - patch.mean(axis=(1,2))

        for y in xrange(H):
            yidx = range(max(y-K/2,0), min(y-K/2+K,H))
            hh = len(yidx)
            for x in xrange(W):
                xidx = range(max(x-K/2,0), min(x-K/2+K,W))
                ww = len(xidx)
                patch = rmeandiff.take(xidx, axis=1).take(yidx, axis=2).reshape(C, hh, ww, N)
                np.square(patch).sum(axis=(1,2), out=routputs[:,x,y,:])

        np.add(r_scale * outputs, 1, out=denoms)
        np.power(denoms, -r_power, out=outputs)
        np.multiply(inputs, outputs, outputs)
        self.eps = 1e-4
        d_inputs = self.be.CUDAMatrix(inputs.copy())
        d_outputs = self.be.CUDAMatrix(np.array(np.empty((C*H*W,N)), dtype=np.single, order='C'))
        d_md = self.be.CUDAMatrix(np.array(np.empty((C*H*W,N)), dtype=np.single, order='C'))
        d_dn = self.be.CUDAMatrix(np.array(np.empty((C*H*W,N)), dtype=np.single, order='C'))
        d_error = self.be.CUDAMatrix(error.copy())
        d_berror = self.be.CUDAMatrix(np.array(np.empty((C*H*W,N)), dtype=np.single, order='C'))
 
        self.be.local_contrast_norm(d_inputs, d_md, d_dn, d_outputs, H, C, K, r_scale, r_power)
        d_outputs.copy_to_host()
        # print routputs[0,:,:,0]
        # print d_outputs.numpy_array.reshape((C,H,W,N))[0,:,:,0]
        # print np.max(np.abs(outputs - d_outputs.numpy_array))
        assert np.max(np.abs(outputs - d_outputs.numpy_array)) < self.eps, "Error in CUDAMatrix.local_contrast_norm exceeded threshold"

        print('\nlocal contrast norm undo:')
        # outputs = d_outputs.numpy_array.copy()
        # routputs = outputs.reshape((C,H,W,N))
        # meandiffs = d_md.numpy_array.copy()
        # rmeandiff = meandiffs.reshape((C,H,W,N))
        # denoms = d_dn.numpy_array.copy()
        # rdenom = denoms.reshape((C,H,W,N))
        prod = np.array(np.empty((C*H*W,N)), dtype=np.single, order='C')
        rprod = prod.reshape((C,H,W,N))
        # -2 x scale x acts x outGrads / denoms
        np.multiply(outputs, -2 * r_scale * r_power * error / denoms, out=outputs)
        for y in xrange(H):
            yidx = range(max(y + K/2 - K + 1, 0), min(y+K/2+1, H))
            hh = len(yidx)
            for x in xrange(W):
                xidx = range(max(x + K/2 - K + 1, 0), min(x+K/2+1, W))
                ww = len(xidx)
                patch = routputs.take(xidx, axis=1).take(yidx, axis=2).reshape(C, hh, ww, N)
                np.sum(patch, axis=(1, 2), out=rprod[:, x, y, :])
        np.multiply(prod, meandiffs, out=prod)
        np.add(prod, error * np.power(denoms, -r_power), out=berror)

        self.be.local_contrast_norm_undo(d_md, d_dn, d_error, d_outputs, d_berror, C, K, r_scale, r_power)
        d_berror.copy_to_host()
        # print 'CC2 Berror\n', d_berror.numpy_array.reshape((C,H,W,N))[0,:,:,0]
        # print 'my berror\n', rberror[0,:,:,0]

        errorg = np.linalg.norm(berror - d_berror.numpy_array)/np.linalg.norm(berror)
        assert errorg < 10**-6, "Error in CUDAMatrix.local_contrast_norm_undo exceeded threshold"
        print('local contrast norm OK')

    @attr('unpool')
    def test_unpool(self):
		np.random.seed(0)
		(N, H, W, C) = (16, 4, 4, 16)   # number of images, ifmshape, channels
		(K, S) = (2, 2)         # Kernel span and stride
		(oH, oW) = (H*2, W*2)
		ifmsize = H*W
		ofmsize = oH*oW
		def f(args): return np.random.randn(*args)*10
		inputs = np.array(f((C*ifmsize, N)), dtype=np.single, order='C')
		outputs = np.array(np.empty((C*ofmsize, N)), dtype=np.single, order='C')
		d_inputs = self.be.CUDAMatrix(inputs.copy())
		d_rinputs = self.be.CUDAMatrix(np.zeros_like(inputs))
		d_outputs = self.be.CUDAMatrix(np.zeros_like(outputs))
		self.be.unpool_forward(smallMat=d_inputs, largeMat=d_outputs, channels=C, sizeX=K, smallX=H, largeX=oH)

		d_outputs.copy_to_host()
		np.set_printoptions(precision=4, linewidth=100, suppress=True)
		print d_outputs.numpy_array.reshape(C, oH, oW, N)[0,:,:,0]
		print inputs.reshape(C,H,W,N)[0,:,:,0]
		outputs =  np.array(f((C*ofmsize, N)), dtype=np.single, order='C')
		d_outputs = self.be.CUDAMatrix(outputs.copy())
		self.be.unpool_backward(largeMat=d_outputs, smallMat=d_rinputs, channels=C, sizeX=K, smallX=H, largeX=oH)
		d_rinputs.copy_to_host()
		print d_rinputs.numpy_array.reshape(C,H,W,N)[0,:,:,0]
		print outputs.reshape(C,oH,oW,N)[0,:,:,0]

    @attr('avg_pool')
    def test_avg_pool(self):
        np.random.seed(0)
        (N, H, W, C) = (128, 4, 4, 16)   # number of images, ifmshape, channels
        (K, S) = (2, 1)         # Kernel span and stride

        (oH, oW) = ((H-K)/S+1, (W-K)/S+1)
        ifmsize = H*W
        ofmsize = oH*oW
        def f(args): return np.random.randn(*args)*10
        def g(args): return np.random.randn(*args)

        inputs = np.array(f((C*ifmsize, N)), dtype=np.single, order='C')
        error = np.array(g((C*ofmsize, N)), dtype=np.single, order='C')
        outputs = np.array(np.empty((C*ofmsize, N)), dtype=np.single, order='C')
        berror = np.array(np.empty((C*ifmsize, N)), dtype=np.single, order='C')

        links = make_local_links(nifm=C, fheight=K, fwidth=K,
                                 ifmheight=H, ifmwidth=W, stride=S)

        outputbuf = np.array(np.empty((ofmsize, N * C)), dtype=np.single, order='C')
        berrorbuf = np.array(np.empty((ifmsize, N * C)), dtype=np.single, order='C')
        prodbuf = np.array(np.empty((K*K, N * C)), dtype=np.single, order='C')

        d_inputs = self.be.CUDAMatrix(inputs.copy())
        d_outputs = self.be.CUDAMatrix(np.zeros_like(outputs))
        d_error = self.be.CUDAMatrix(error.copy())
        d_berror = self.be.CUDAMatrix(np.zeros_like(berror))

        #FPROP ########################
        print '\navg pool:'
        rinputs = hstack_maps(inputs, C)
        for dst in xrange(ofmsize):
            rf = rinputs.take(links[dst], axis=0)
            outputbuf[dst, :] = rf.mean(axis=0)
        outputs[:] = vstack_maps(outputbuf, C)

        self.be.avg_pool(imgs=d_inputs, target=d_outputs, channels=C, sizeX=K,
                         paddingStart=0, moduleStride=S, numModulesX=oH)
        d_outputs.copy_to_host()
        assert np.max(np.abs(outputs - d_outputs.numpy_array)) < 10**-5, "Error in CUDAMatrix.avg_pool exceeded threshold"

        #BPROP #########################
        print '\navg pool undo:'
        berrorbuf.fill(0.0)
        error /= K * K
        rerror = hstack_maps(error, C)
        for dst in xrange(ofmsize):
            berrorbuf[links[dst], :] += rerror[dst, :]
        berror[:] = vstack_maps(berrorbuf, C)

        self.be.avg_pool_undo(avgGrads=d_error, target=d_berror, sizeX=K,
                              paddingStart=0, moduleStride=S, numModulesX=oW,
                              imgSizeX=W)
        d_berror.copy_to_host()
        assert np.max(np.abs(berror - d_berror.numpy_array)) < 10**-5, "Error in CUDAMatrix.avg_pool_undo exceeded threshold"

        #FPROP ########################
        print '\nl2 pool:'
        rinputs = hstack_maps(inputs, C)
        for dst in xrange(ofmsize):
            rf = rinputs.take(links[dst], axis=0)
            outputbuf[dst, :] = np.sum(np.abs(rf)**2, axis=0)**(1.0/2)/(K*K)
        outputs[:] = vstack_maps(outputbuf, C)

        self.be.l2_pool(imgs=d_inputs, target=d_outputs, channels=C, sizeX=K,
                         paddingStart=0, moduleStride=S, numModulesX=oH)
        d_outputs.copy_to_host()

        assert np.max(np.abs(outputs - d_outputs.numpy_array)) < 10**-6, "Error in CUDAMatrix.l2_pool exceeded threshold"

        #BPROP ########################
        print '\nl2 pool undo:'
        routputs = hstack_maps(outputs, C)
        rerror = hstack_maps(error, C)
        berrorbuf.fill(0.0)
        for dst in xrange(ofmsize):
            inds = links[dst]
            rf = rinputs.take(inds, axis=0)
            denom = routputs[dst, :].copy()
            # If the L2 norm is zero, the entire receptive field must be
            # zeros. In that case, we set the L2 norm to 1 before using
            # it to normalize the receptive field.
            denom[denom == 0] = 1
            np.divide(rf, denom, out=rf)
            np.multiply(
                rerror[dst:(dst + 1), :].repeat(K*K, axis=0), rf, out=prodbuf)
            berrorbuf[inds, :] += prodbuf
        berror[:] = vstack_maps(berrorbuf, C)
        berror = berror/(K*K)

        self.be.l2_pool_undo(imgs=d_inputs, l2Grads=d_error, l2Acts=d_outputs,
                             target=d_berror, sizeX=K, paddingStart=0,
                             moduleStride=S, numModulesX=oH)
        d_berror.copy_to_host()
        assert np.max(np.abs(berror - d_berror.numpy_array)) < 10**-6, "Error in CUDAMatrix.l2_pool_undo exceeded threshold"

    @attr('cmap_max_pool')
    def test_cmap_max_pool(self):
        print "TODO"

    @attr('clip')
    def test_clip(self):
        print '\nclip:'
        np.random.seed(0)
        m = 4
        k = 3
        a = np.array(np.random.randn(m, k)*10, dtype=np.float32, order='C')
        m1 = self.be.CUDAMatrix(a)
        m2 = self.be.CUDAMatrix(np.zeros_like(a))
        self.be.clip_range(m1, -3.0, 3.0, target=m2)
        m2.copy_to_host()
        ac = np.clip(a, -3., 3.)

        assert np.max(np.abs(ac - m2.numpy_array)) < 10**-5, "Error in CUDAMatrix.clip exceeded threshold"

    @attr('dropout')
    def test_dropout(self):
        print '\ndropout:'
        np.random.seed(0)
        self.be.cudanet_init_random()
        m = 4
        k = 3
        a_u = np.array(np.random.uniform(size=(m, k)), dtype=np.float32, order='C')
        m1 = self.be.CUDAMatrix(np.zeros_like(a_u))
        m1.randomize_uniform_thresh(keepthresh=0.5)
        m1.copy_to_host()
        print m1.numpy_array
        self.be.cudanet_destroy_random()

    @attr('maxscal')
    def test_maximum_scalar(self):
        print('\nmax:')
        np.random.seed(0)
        m = 4
        k = 3
        a = np.array(np.random.randn(m, k)*10, dtype=np.float32, order='C')

        m1 = self.be.CUDAMatrix(a)
        hmax = np.maximum(a, 0)
        self.be.maximum_scalar(m1, 0)
        m1.copy_to_host()
        assert np.max(np.abs(hmax - m1.numpy_array)) < 10**-5, "Error in CUDAMatrix.maximum_scalar exceeded threshold"
        print('maximum scalar OK')

    @attr('maximum')
    def test_maximum(self):
        print('\nmax:')
        np.random.seed(0)
        m = 4
        k = 3
        a = np.array(np.random.randn(m, k)*10, dtype=np.float32, order='C')
        b = np.array(np.random.randn(m, k)*10, dtype=np.float32, order='C')

        m1 = self.be.CUDAMatrix(a)
        m2 = self.be.CUDAMatrix(b)

        m1.copy_to_host()
        hmax = np.maximum(a, b)
        self.be.maximum(m1, m2)
        m1.copy_to_host()
        assert np.max(np.abs(hmax - m1.numpy_array)) < 10**-5, "Error in CUDAMatrix.maximum exceeded threshold"
        print('maximum OK')

    def test_max(self):
        print('\nmax:')
        np.random.seed(0)
        m = 4
        k = 3
        a = np.array(np.random.randn(m, k)*10, dtype=np.float32, order='C')

        hval = a.max()
        m1 = self.be.CUDAMatrix(a)
        m2 = m1.max(axis=0)
        m3 = m2.max(axis=1)
        m4 = m1.max(axis=None)
        m3.copy_to_host()
        m4.copy_to_host()
        dval = m3.numpy_array[0][0]
        dval2 = m4.numpy_array[0][0]
        print('hval', hval, 'dval', dval, 'dval2', dval2)

        assert np.abs(hval - dval) < 10**-2, "Error in max exceeded threshold"
        assert np.abs(hval - dval2) < 10**-2, "Error in maxGlobal exceeded threshold"

        print('max OK')


    def test_where(self):
        print('\nwhere:')
        m = 256
        n = 128
        a = np.array(np.random.randn(m, n)*10, dtype=np.float32, order='C')
        z = np.zeros_like(a)
        res = np.where(a > 0, a, z);

        a_d = self.be.CUDAMatrix(a)
        z_d = self.be.CUDAMatrix(z)
        res_d = self.be.empty(a_d.shape)
        a_d.greater_than(0,  res_d)
        self.be.where(res_d, a_d, z_d)
        assert np.abs(res-res_d.asarray()).max() < 1e-2, "Error in self.be.where"

        print('where OK')

    @attr('softmax')
    def test_softmax(self):
        print 'softmax\n'
        m = 10
        n = 128
        np.random.seed(0)
        a = np.array(np.random.randn(m, n)*10, dtype=np.float32, order='C')
        da = self.be.CUDAMatrix(a.copy())
        dc2 = self.be.CUDAMatrix(np.zeros_like(a))

        #CPU softmax
        a = np.exp(a - np.max(a, axis=0))
        a = a/np.sum(a, axis=0)

        dc = self.be.softmax(da, axis=0)
        self.be.softmax(da, target=dc2, axis=0)
        dc.copy_to_host()
        dc2.copy_to_host()
        errorv = np.linalg.norm(a - dc.numpy_array)
        errorv2 = np.linalg.norm(a - dc2.numpy_array)

        assert errorv < 1e-5, "Error in softmax"
        assert errorv2 < 1e-5, "Error in softmax"

        print 'softmax grad\n'
        herr = np.array(np.random.randn(m, n), dtype=np.float32, order='C')
        derr = self.be.CUDAMatrix(herr.copy())
        hY = a
        dY = dc

        dGrad = self.be.softmax_grad(dY, derr)
        hGrad = (herr - np.einsum('ij,ji->i', herr.T, hY))*hY
        dGrad.copy_to_host()

        errorg = np.linalg.norm(hGrad - dGrad.numpy_array)
        assert errorg < 1e-5, "Error in softmax gradient"

    @attr('crossent')
    def test_crossent(self):
        np.random.seed(0)
        m = 10
        n = 128

        # Make a random set of labels (128 points, each with 10 outputs, only 1 correct)
        idx = np.int32(np.array(np.random.uniform(size=(1,n)), dtype=np.float32, order='C')*(m-1))
        idx = np.array(idx, dtype= np.int32)
        lbls = np.zeros((m,n), dtype=np.float32, order='C')
        lbls[idx, np.arange(n)] = 1.

        # Make a bunch of probabilities for each output
        probs = np.array(np.random.uniform(size=(m,n)), dtype=np.float32, order='C')
        probs = probs/np.sum(probs,axis=0)


        dProbs = self.be.CUDAMatrix(probs.copy())
        dLbls = self.be.CUDAMatrix(lbls.copy())

        cEnt = np.sum(lbls*(np.log(probs)), axis=0)

        dcEnt = self.be.crossent_cost(dLbls, dProbs)
        dcEnt.copy_to_host()
        errorv = np.linalg.norm(cEnt - dcEnt.numpy_array)
        assert errorv < 1e-5, "Error in crossent"

        dcEntG = self.be.crossent_cost_grad(dLbls, dProbs)
        dcEntG.copy_to_host()

        cEntG = lbls/probs
        errorv = np.linalg.norm(cEntG - dcEntG.numpy_array)
        assert errorv < 1e-5, "Error in crossentGrad"


    @attr('weight_norm')
    def test_weight_norm(self):
        np.random.seed(0)
        m = 10
        n = 128
        wMax = 5.
        a = np.array(np.random.randn(m, n), dtype=np.float32, order='C')
        a[:,100:] = a[:,100:]*10

        dwts = self.be.CUDAMatrix(a.copy())
        dwts_norm = self.be.weight_norm_along_axis(dwts, axis=0, norm=wMax)
        dwts_norm.copy_to_host()

        # cpu version
        nn = np.sum(a*a, axis=0)
        nnrecip = np.where(nn > wMax*wMax, wMax/np.sqrt(nn), 1)
        wts_norm = a*nnrecip

        errorv = np.linalg.norm(wts_norm - dwts_norm.numpy_array)
        assert errorv < 1e-5, "Error in crossentGrad"


        dwts_norm = self.be.weight_norm_along_axis(dwts, axis=1, norm=wMax)
        dwts_norm.copy_to_host()

        # cpu version
        nn = np.sum(a*a, axis=1,keepdims=True)
        nnrecip = np.where(nn > wMax*wMax, wMax/np.sqrt(nn), 1)
        wts_norm = a*nnrecip

        errorv = np.linalg.norm(wts_norm - dwts_norm.numpy_array)
        assert errorv < 1e-5, "Error in crossentGrad"

    @attr('allminmax')
    def test_minmax(self):
        m = 256
        n = 128
        for op in 'min', 'max', 'argmax', 'argmin':
            print('\n%s' % (op))
            for sign in (1, -1):
                a = np.array(np.random.randn(m, n)*10*sign, dtype=np.float32, order='C')
                t0 = np.array(np.random.rand(1, n)*10, dtype=np.float32, order='C')
                t1 = np.array(np.random.rand(m, 1)*10, dtype=np.float32, order='C')

                r0 = np.atleast_2d(getattr(a, op)(0))
                r1 = np.atleast_2d(getattr(a, op)(1))

                da = self.be.CUDAMatrix(a)
                dr10 = self.be.CUDAMatrix(t0)
                dr11 = self.be.CUDAMatrix(t1)

                getattr(da, op)(axis = 0, target = dr10)
                getattr(da, op)(axis = 1, target = dr11)
                dr20 = getattr(da, op)(axis = 0)
                dr21 = getattr(da, op)(axis = 1)

                dr10.copy_to_host()
                dr11.copy_to_host()
                dr20.copy_to_host()
                dr21.copy_to_host()

                assert np.max(np.abs(r0 - dr10.numpy_array)) < 10**-4, "Error in CUDAMatrix.%s exceeded threshold" % op
                assert np.max(np.abs(r1 - dr11.numpy_array.T)) < 10**-4, "Error in CUDAMatrix.%s exceeded threshold" % op
                assert np.max(np.abs(r0 - dr20.numpy_array)) < 10**-4, "Error in CUDAMatrix.%s exceeded threshold" % op
                assert np.max(np.abs(r1 - dr21.numpy_array.T)) < 10**-4, "Error in CUDAMatrix.%s exceeded threshold" % op
            print('%s OK' % (op))
