#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <ur_kinematics/ur_kin.h>


static PyObject* forward(PyObject* self, PyObject* args)
{
    PyArrayObject* q_array;
    PyArrayObject* T_array;

    if (!PyArg_ParseTuple(args, "O!O!",
            &PyArray_Type, &q_array,
            &PyArray_Type, &T_array))
        return NULL;

    double* q = (double *)PyArray_DATA(q_array);
    double* T = (double *)PyArray_DATA(T_array);

    ur_kinematics::forward(q, T);
    return PyLong_FromLong(1);
}

static PyObject* forward_all(PyObject* self, PyObject* args)
{
    PyArrayObject* q_array;
    PyArrayObject* T_array[6];

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!",
            &PyArray_Type, &q_array,
            &PyArray_Type, &T_array[0],
            &PyArray_Type, &T_array[1],
            &PyArray_Type, &T_array[2],
            &PyArray_Type, &T_array[3],
            &PyArray_Type, &T_array[4],
            &PyArray_Type, &T_array[5]))
        return NULL;

    double* q = (double *)PyArray_DATA(q_array);
    double* T[6];
    for( int i = 0; i< 6; ++i)
        T[i] = (double *)PyArray_DATA(T_array[i]);

    ur_kinematics::forward_all(q, T[0], T[1], T[2], T[3], T[4], T[5]);
    return PyLong_FromLong(1);
}

static PyObject* inverse(PyObject* self, PyObject* args)
{
    PyArrayObject* T_array;
    PyArrayObject* q_array;
    double q6_des = 0.0;

    if (!PyArg_ParseTuple(args, "O!O!|d",
            &PyArray_Type, &T_array,
            &PyArray_Type, &q_array,
            &q6_des))
        return NULL;

    double* T = (double *)PyArray_DATA(T_array);
    double* q = (double *)PyArray_DATA(q_array);

    int n = ur_kinematics::inverse(T, q, q6_des);
    return PyLong_FromLong(n);
}


static PyMethodDef ModuleMethods[] =
{
     {"forward", forward, METH_VARARGS, "compute forward kinematics"},
     {"forward_all", forward_all, METH_VARARGS, "compute forward kinematics for all joints"},
     {"inverse", inverse, METH_VARARGS, "compute inverse kinematics"},
     {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cModPyDem =
{
    PyModuleDef_HEAD_INIT,
    "ur5_kinematics",
    "Provides forward and inverse kinematics for Universal Robots designs.",
    -1,
    ModuleMethods
};

PyMODINIT_FUNC
PyInit_ur5_kinematics(void)
{
    PyObject* module = PyModule_Create(&cModPyDem);
    import_array();
    return module;
}
