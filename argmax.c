
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include<Python.h>
#include<numpy/arrayobject.h>
#include<emmintrin.h>


static npy_intp _bool_argmax(npy_bool* items, npy_intp n, npy_intp stride) {
    // Function invoked when the maximum value search must be performed forward
    npy_intp i = 0;

    // Algorithm when stride value is 1 (we can apply SSE2 optimizations)
    if(stride == 1) {
#ifdef __SSE2__
        const __m128i zero = _mm_setzero_si128();
        for(; i<n-(n % 32); i+=32) {
            __m128i a = _mm_loadu_si128((__m128i*)&items[i]);
            __m128i b = _mm_loadu_si128((__m128i*)&items[i+16]);
            a = _mm_cmpeq_epi8(a, zero);
            b = _mm_cmpeq_epi8(b, zero);
            if (_mm_movemask_epi8(_mm_min_epu8(a, b)) != 0xFFFF)
                break;
        }
#endif
        for(; i<n; i++) {
            if(items[i])
                return i;
        }
    }
    else {
        // stride value > 1
        for(; i<n; i+=stride) {
            if(items[i])
                return i;
        }
    }

    return 0;
}

static npy_intp _reversed_bool_argmax(npy_bool* items, npy_intp n, npy_intp stride) {
    // Function invoked when the maximum value search must be performed backwards
    npy_intp i = n-1;

    // Algorithm when stride value is 1 (we can apply SSE2 optimizations)
    if(stride == 1) {
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
        if(!(i || items[0]))
            i = n-1;
    }
    else {
        // Algorithm without optimizations (stride value > 1)
        for(; i>0; i-=stride) {
            if(items[i])
                break;
        }
        if(i < 0 || !(i || items[0]))
            i = n-1;
    }

    return i;
}



static PyObject* bool_argmax(PyObject* self, PyObject* args) {
    PyArrayObject *in;
    npy_bool *items;
    npy_intp n, stride, i;
    Py_ssize_t nargs;


    // Only 2 arguments accepted: The input array and stride size (default to 1)
    nargs = PyTuple_Size(args);
    if(nargs == 0) {
        PyErr_SetString(PyExc_ValueError, "You must indicate the input array");
        return NULL;
    }

    if(nargs > 2) {
        PyErr_SetString(PyExc_ValueError, "Invalid number of positional arguments");
        return NULL;
    }

    // Convert the first argument to a numpy array (contiguous and aligned in memory)
    in = (PyArrayObject*)PyArray_FromAny(PyTuple_GET_ITEM(args, 0), NULL, 0, 0,  NPY_ARRAY_IN_ARRAY, NULL);

    // Get the stride value
    stride = 1;
    if(nargs == 2) {
        stride = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, 1));
        if(PyErr_Occurred() || stride == 0) {
            PyErr_SetString(PyExc_ValueError, "Stride value must be a non-zero number");
            return NULL;
        }
    }

    // Only 1-dimensional arrays accepted
    if(PyArray_NDIM(in) != 1) {
        PyErr_SetString(PyExc_ValueError, "Only 1D arrays are valid");
        return NULL;
    }

    // Only boolean arrays accepted
    if(!PyTypeNum_ISBOOL(PyArray_TYPE(in))) {
        PyErr_SetString(PyExc_ValueError, "Only non empty boolean arrays are valid");
        return NULL;
    }

    // Get the items of the array and its length
    items = (npy_bool*)PyArray_DATA(in);
    n = PyArray_DIM(in, 0);

    // The stride argument must be lower or equal than the number of items
    if(stride > n || stride < -n) {
        PyErr_SetString(PyExc_ValueError, "Stride value must be lower or equal than the number of items in the array");
        return NULL;
    }

    // Apply a different algorithm depending on the stride sign.
    if(stride > 0)
        i = _bool_argmax(items, n, stride);
    else
        i = _reversed_bool_argmax(items, n, -stride); // swap the sign of the stride

    // Return the result
    return PyLong_FromSize_t(i);
}


static char docstring[] =
"bool_argmax calculates the index of an item in the given non empty 1D boolean array with maximum value."
"the 2nd argument (stride) must be a number different than zero, and controls how the maximum value will be searched in the array."
"The sign of the stride value controls how the search is performed (from the beginning to the end of the array if its positive or backwards otherwise)."
"The abolute value can be used to skip some items in the search: A value of 2 will indicate to search the maximum between elements with even indices";


static PyMethodDef methods[] = {
    {"bool_argmax", bool_argmax, METH_VARARGS, docstring}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "numpy_argmax_ext",
    "Additional helper methods to improve numpy.argmax performance for 1D boolean arrays with strides",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_numpy_bool_argmax_ext(void)
{
    import_array();
    return PyModule_Create(&module);
}
