cimport cython
cimport numpy as np
import numpy as np

from dipy.tracking.propspeed cimport _propagation_direction
from dipy.tracking.local.direction_getter cimport DirectionGetter

from libc.math cimport round


def make_nd(array, N):
    """Makes an array that's less than Nd - Nd

    We need this because numpy 1.6 does not return a "c contiguous array"
    when you call ``array(a, order='c', ndmin=N)``
    """
    if array.ndim > N:
        raise ValueError()
    new_shape = (1,) * (N - array.ndim) + array.shape
    return array.reshape(new_shape)


cdef class PAMDirectionGetter(DirectionGetter):

    cdef:
        public double qa_thr, ang_thr, total_weight
        public double[:, :, :, ::1] _qa, _ind
        public double[:, ::1] _odf_vertices
        int initialized

    def __cinit__(self):
        self.qa_thr = 0.0239
        self.ang_thr = 60
        self.total_weight = .5

    def _initialize(self):
        if self.peak_values.shape != self.peak_indices.shape:
            msg = "shapes of peak_values and peak_indices do not match"
            raise ValueError(msg)
        self._qa = make_nd(np.array(self.peak_values, copy=False,
                                   dtype='double', order='C'), 4)
        self._ind = make_nd(np.array(self.peak_indices, copy=False,
                                    dtype='double', order='C'), 4)
        self._odf_vertices = np.array(self.sphere.vertices, copy=False,
                                      dtype='double', order='C')

        self.initialized = 1

    def initial_direction(self, double[::1] point):
        """The best starting directions for fiber tracking from point

        All the valid peaks in the voxel closest to point are returned as
        initial directions.

        """
        cdef:
            np.npy_intp numpeaks, i
            np.npy_intp ijk[3]

        # ijk is the closest voxel to point
        for i in range(3):
            ijk[i] = <np.npy_intp> round(point[i])
            if ijk[i] < 0 or ijk[i] >= self._ind.shape[i]:
                raise IndexError("point outside data")

        # Check to see how many peaks were found in the voxel
        for numpeaks in range(self._ind.shape[3]):
            if self._ind[ijk[0], ijk[1], ijk[2], numpeaks] < 0:
                break

        # Create directions array and copy peak directions from vertices
        res = np.empty((numpeaks, 3))
        for i in range(numpeaks):
            peak_index = self._ind[ijk[0], ijk[1], ijk[2], i]
            res[i, :] = self._odf_vertices[<np.npy_intp> peak_index, :]

        return res


    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int get_direction(self, double[::1] point, double[::1] direction):
        """Interpolate closest peaks to direction from voxels neighboring point

        Update direction and return 0 if successful. If no tracking direction
        could be found, return 1.

        """
        cdef:
            np.npy_intp s
            double newdirection[3]
            np.npy_intp qa_shape[4]
            np.npy_intp qa_strides[4]

        if not self.initialized:
            return -1

        for i in range(4):
            qa_shape[i] = self._qa.shape[i]
            qa_strides[i] = self._qa.strides[i]

        s = _propagation_direction(&point[0], &direction[0],
                                   &self._qa[0, 0, 0, 0],
                                   &self._ind[0, 0, 0, 0],
                                   &self._odf_vertices[0, 0], self.qa_thr,
                                   self.ang_thr, qa_shape, qa_strides,
                                   newdirection, self.total_weight)
        if s:
            for i in range(3):
                direction[i] = newdirection[i]
            return 0
        else:
            return 1
