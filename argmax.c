
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include<Python.h>
#include<numpy/arrayobject.h>
#include<emmintrin.h>




static PyObject* reversed_bool_argmax(PyObject* self, PyObject* args) {
    PyObject *arg;
    PyArrayObject *in;
    npy_bool *items;
    npy_intp n;

    // Only 1 argument accepted
    if(PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_ValueError, "You must indicate one and only one argument");
        return NULL;
    }

    // Get the first argument
    arg = PyTuple_GET_ITEM(args, 0);

    // Convert the argument to a numpy array (contiguous and aligned in memory)
    in = (PyArrayObject*)PyArray_FromAny(arg, NULL, 0, 0,  NPY_ARRAY_IN_ARRAY, NULL);

    // Only 1 or 0-dimensional arrays accepted
    if(PyArray_NDIM(in) > 1) {
        PyErr_SetString(PyExc_ValueError, "Only arrays with one or zero dimensions are valid");
        return NULL;
    }

    // Only boolean arrays accepted
    if(!PyTypeNum_ISBOOL(PyArray_TYPE(in))) {
        PyErr_SetString(PyExc_ValueError, "Only non empty boolean arrays are valid");
        return NULL;
    }

    // If array is 0 dimensional, return 0 directly
    if(PyArray_NDIM(in) == 0) {
        return PyLong_FromSize_t(0);
    }

    // Get the items of the array and its length
    items = (npy_bool*)PyArray_DATA(in);
    n = PyArray_DIM(in, 0);

    // Here is the algorithm.
    // POSTCONDITION: i will index will point to an element with maximum value on the array
    // starting from the end (If all items are set to False, it will be 0)

    npy_intp i = n-1;

#ifdef __SSE2__
    const __m128i zero = _mm_setzero_si128();
    for(; i>n%32; i-=32) {
        __m128i a = _mm_loadu_si128((__m128i*)&items[i-31]);
        __m128i b = _mm_loadu_si128((__m128i*)&items[i-15]);
        a = _mm_cmpeq_epi8(a, zero);
        b = _mm_cmpeq_epi8(b, zero);
        if (_mm_movemask_epi8(_mm_min_epu8(a, b)) != 0xFFFF)
            break;
    }
#endif
    for(; i>0; i--) {
        if(items[i])
            break;
    }
    if(!(i>0 || items[0]))
        i = n-1;

    return PyLong_FromSize_t(i);
}


static PyMethodDef methods[] = {
    {"reversed_bool_argmax", reversed_bool_argmax, METH_VARARGS, "Same as numpy.argmax for 1D boolean arrays but it starts search the maximum value at the end and goes backwards"}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "numpy_argmax_ext",
    "Additional helper methods to improve numpy.argmax performance for 1D boolean arrays with inverted strides",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_numpy_bool_argmax_ext(void)
{
    import_array();
    return PyModule_Create(&module);
}
