#ifndef DEEPONET_H
#define DEEPONET_H
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstdio>

static PyObject* pModule = NULL;
static PyObject* pFunc;
static PyObject* pFunc_vmap;
static int n_scalar_in = 10;
static int n_scalar_out = 60; // Same as total number of species (including N2)
static int max_ncells = 0;
static int nspec_tot_recnet =
  162; // Maximum scalars that can be reconstructed + representative scalars

static double* u0_subarray;
static double* sie_subarray;

static int
udf_py_init()
{
  PyObject* pName;
  // dlopen("libpython3.8.so.1.0", RTLD_LAZY | RTLD_GLOBAL);
  //  Initialize the Python interpreter
  Py_Initialize();
  // Initialize the NumPy module
  import_array();

  // Set PYTHONPATH to the current directory
  PyRun_SimpleString("import sys\n");
  PyRun_SimpleString("sys.path.append(\".\")\n");

  // Import the Python module
  pName = PyUnicode_DecodeFSDefault("DeepONet_Pred_C");
  pModule = PyImport_Import(pName);

  pFunc = PyObject_GetAttrString(pModule, "predict_from_c");
  pFunc_vmap = PyObject_GetAttrString(pModule, "predict_from_c_vmap");

  Py_DECREF(pName);

  if (pModule != NULL) {
    printf("Python module imported successfully.\n");

    if (pFunc && PyCallable_Check(pFunc)) {

      printf("The Python function is callable.\n");
    }

    else {
      printf("The Python function is not callable.\n");
    }

  }

  else {
    printf("Failed to import the Python module.\n");
  }

  return 0;
}

static int
main_check(double u0_star[], double y_star, double equiv, double ret_vals[])

{

  PyObject* pArgs;

  // Get the reference to the Python function
  // pFunc = PyObject_GetAttrString(pModule, "predict_from_c");

  PyArrayObject* u0_star_array;
  PyObject* u0_star_value;
  PyObject* pValue;
  PyObject* pItem;
  PyObject* y_star_value;
  PyObject* equiv_value;

  PyArrayObject* numpy_array;
  double* c_array;
  npy_intp* shape;

  int i;

  // Create the NumPy array for u0_star
  npy_intp dims[1] = {n_scalar_in}; // Shape of the array
  u0_star_array =
    (PyArrayObject*)PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, u0_star);
  // u0_star_value =
  // Py_BuildValue("(dddddd)",u0_star[0],u0_star[1],u0_star[2],u0_star[3],u0_star[4],u0_star[5]);

  // Create the arguments for the Python function
  pArgs = PyTuple_New(3);
  PyTuple_SetItem(pArgs, 0, (PyObject*)u0_star_array);
  // PyTuple_SetItem(pArgs, 0, u0_star_value);

  // Create the input values for y_star and equiv
  y_star_value = PyFloat_FromDouble(y_star);
  equiv_value = PyFloat_FromDouble(equiv);
  PyTuple_SetItem(pArgs, 1, y_star_value);
  PyTuple_SetItem(pArgs, 2, equiv_value);

  // Call the Python function
  pValue = PyObject_CallObject(pFunc, pArgs);

  // Dereference the PyTuple and its contents
  Py_DECREF(pArgs);
  Py_DECREF(y_star_value);
  Py_DECREF(equiv_value);

  numpy_array =
    (PyArrayObject*)PyArray_FROM_OTF(pValue, NPY_DOUBLE, NPY_IN_ARRAY);

  if (!numpy_array) {
    printf("Numpy Array not converted");
    return -1;
  }

  // Get a pointer to the data in the NumPy array
  c_array = (double*)PyArray_DATA(numpy_array);

  shape = PyArray_DIMS(numpy_array);

  for (i = 0; i < n_scalar_out; i++) {
    ret_vals[i] = c_array[i];
  }

  // for (i=0; i<n_species; i++)
  // {
  //  pItem=PyTuple_GetItem(pValue, i);
  //  ret_vals[i]= PyFloat_AsDouble(pItem);
  //  }
  Py_DECREF(numpy_array);
  Py_DECREF(pValue);

  // Print the returned values
  // printf("Returned value 1: %f\n", ret_vals[0]);
  // printf("Returned value 2: %f\n", ret_vals[1]);

  return 0;
}

static int
udf_py_fin()
{
  if (pModule != NULL) {

    Py_DECREF(pModule);
    printf("The Python module is dereferenced.\n");

    if (pFunc != NULL) {
      Py_DECREF(pFunc);
      printf("The Python function is dereferenced.\n");
    }

    if (pFunc_vmap != NULL) {
      Py_DECREF(pFunc_vmap);
      printf("The Python function is dereferenced.\n");
    }
  }
  free(u0_subarray);
  free(sie_subarray);
  Py_Finalize();
  return 0;
}

static int
MF_DeepONet(
  double local_dt, double mass_fractions[], double sie, double* temperature)

{

  double param;

  double mf_sum;
  int i, j;

  param = 1.0;

  PyObject* pArgs;
  PyObject* sie_value;
  PyArrayObject* u0_star_array;
  PyObject* pValue;
  // PyObject* pItem;
  PyObject* y_star_value;
  PyObject* param_value;

  PyArrayObject* numpy_array;
  double* c_array;
  npy_intp* shape;

  // Create the NumPy array for u0_star
  npy_intp dims[1] = {n_scalar_in - 1}; // Shape of the array
  u0_star_array = (PyArrayObject*)PyArray_SimpleNewFromData(
    1, dims, NPY_DOUBLE, mass_fractions);

  // Create the arguments for the Python function
  pArgs = PyTuple_New(4);
  PyTuple_SetItem(pArgs, 0, (PyObject*)u0_star_array);

  // Create the input values for y_star and param and internal energy
  y_star_value = PyFloat_FromDouble(local_dt);
  param_value = PyFloat_FromDouble(param);
  sie_value = PyFloat_FromDouble(sie);

  PyTuple_SetItem(pArgs, 1, y_star_value);
  PyTuple_SetItem(pArgs, 2, param_value);
  PyTuple_SetItem(pArgs, 3, sie_value);

  // Call the Python function
  pValue = PyObject_CallObject(pFunc, pArgs);

  // Dereference the PyTuple and its contents
  Py_DECREF(pArgs);
  Py_DECREF(y_star_value);
  Py_DECREF(param_value);
  Py_DECREF(sie_value);

  numpy_array =
    (PyArrayObject*)PyArray_FROM_OTF(pValue, NPY_DOUBLE, NPY_IN_ARRAY);

  if (!numpy_array) {
    printf("Numpy Array not converted");
    return -1;
  }

  // Get a pointer to the data in the NumPy array
  c_array = (double*)PyArray_DATA(numpy_array);

  shape = PyArray_DIMS(numpy_array);

  mf_sum = 0.0;
  for (i = 1; i < n_scalar_out; i++) {
    mass_fractions[i - 1] = c_array[i];
    if (mass_fractions[i - 1] < 0.0) {
      mass_fractions[i - 1] = 0.0;
    } // Zeroing negative values
    mf_sum = mf_sum + mass_fractions[i - 1];
  }

  mass_fractions[n_scalar_out - 1] = 1.0 - mf_sum;

  Py_DECREF(numpy_array);
  Py_DECREF(pValue);

  // Don't update when internal energy is supplied
  //  *temperature = c_array[0];

  return 1;
}

static void
check_deeponet()

{

  double u0_star[n_scalar_out];
  double y_star, equiv, sie;
  double temperature;

  clock_t start_t, end_t;
  double timetaken;
  int i, j, nstep;

  printf("The Call from UDF to C_DeepONet.\n");

  memset(u0_star, 0, n_scalar_out * sizeof(double));

  // Initial Condition
  sie = 1246205.3039284516 - 2E6;
  u0_star[0] = 1.23015463e-01;
  u0_star[1] = 3.59179936e-02;
  u0_star[2] = 1.45457045e-03;
  u0_star[3] = 5.54114394e-02;
  u0_star[4] = 2.48897239e-04;
  u0_star[5] = 5.17131877e-04;
  u0_star[6] = 3.58152378e-04;
  u0_star[7] = 8.22510570e-02;
  u0_star[8] = 2.72807060e-03;
  u0_star[9] = 0.0;

  y_star = 1E-6; // In ms
  equiv = 1.0;

  // u0 =
  // np.array([1.1400000e+03, 6.0910187e-14, 5.6240503e-12, 5.5166412e-02, 1.3082714e-18,
  // 2.9821038e-14])

  start_t = clock();

  for (i = 0; i < 1; i++) {

    nstep = MF_DeepONet(y_star, u0_star, sie, &temperature);
  }

  end_t = clock();

  printf(
    "Total time taken: %f\n", ((double)(end_t - start_t)) / CLOCKS_PER_SEC);

  // Print the returned values
  printf("Returned value 1: %f\n", u0_star[0]);
  printf("Returned value 2: %f\n", u0_star[1]);
  printf("Returned value 3: %f\n", u0_star[2]);
  printf("Returned value 4 (N2): %f\n", u0_star[n_scalar_out - 1]);
  printf("Returned Temperature: %f\n", temperature);
  printf("Returned steps : %d\n", nstep);
}

static int
MF_DeepONet_VMAP(
  double local_dt,
  double mass_fractions[],
  double sie[],
  double temperature[],
  int num_rows,
  int tot_num_spec)

{

  double param = 0.0; // this needs to be one of the input to this function

  double mf_sum;
  int i, j;

  param = 1.0;

  PyObject* pArgs;
  PyArrayObject* sie_array;
  PyArrayObject* u0_star_array;

  PyObject* pValue;
  // PyObject* pItem;
  PyObject* y_star_value;
  PyObject* param_value;

  PyArrayObject* numpy_array;
  double* c_array;
  npy_intp* shape;

  if (tot_num_spec != n_scalar_out) {
    printf("Don't proceed, mass-fraction array length mismatch\n");
  }

  if (num_rows > max_ncells)

  {
    if (max_ncells > 0) {
      free(u0_subarray);
      free(sie_subarray);
    }

    max_ncells = (int)(10.0 * (float)num_rows);

    u0_subarray =
      (double*)malloc(max_ncells * (n_scalar_in - 1) * sizeof(double));
    sie_subarray = (double*)malloc(max_ncells * sizeof(double));
  }

  for (i = 0; i < num_rows; i++) {
    for (j = 0; j < (n_scalar_in - 1); j++) {

      u0_subarray[i * (n_scalar_in - 1) + j] =
        mass_fractions[i * n_scalar_out + j];
    }
    sie_subarray[i] = sie[i];
  }

  // Create the NumPy array for u0_star
  npy_intp dims[2] = {max_ncells, n_scalar_in - 1}; // Shape of the array
  u0_star_array =
    (PyArrayObject*)PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, u0_subarray);

  npy_intp sie_dims[1] = {max_ncells};
  sie_array = (PyArrayObject*)PyArray_SimpleNewFromData(
    1, sie_dims, NPY_DOUBLE, sie_subarray);

  // Create the arguments for the Python function
  pArgs = PyTuple_New(4);
  PyTuple_SetItem(pArgs, 0, (PyObject*)u0_star_array);

  // Create the input values for y_star and param and internal energy
  y_star_value = PyFloat_FromDouble(local_dt);
  param_value = PyFloat_FromDouble(param);
  // sie_value = PyFloat_FromDouble(sie);

  PyTuple_SetItem(pArgs, 1, y_star_value);
  PyTuple_SetItem(pArgs, 2, param_value);

  PyTuple_SetItem(pArgs, 3, (PyObject*)sie_array);

  // Call the Python function
  pValue = PyObject_CallObject(pFunc_vmap, pArgs);

  // Dereference the PyTuple and its contents
  Py_DECREF(pArgs);
  Py_DECREF(y_star_value);
  Py_DECREF(param_value);
  // Py_DECREF(sie_value);

  numpy_array =
    (PyArrayObject*)PyArray_FROM_OTF(pValue, NPY_DOUBLE, NPY_IN_ARRAY);

  if (!numpy_array) {
    printf("Numpy Array not converted");
    return -1;
  }

  // Check if the array is two-dimensional
  if (PyArray_NDIM(numpy_array) != 2) {
    printf("Expected a 2D Numpy array\n");
    Py_DECREF(numpy_array);
    return -1;
  }

  // Get a pointer to the data in the NumPy array
  c_array = (double*)PyArray_DATA(numpy_array);

  shape = PyArray_DIMS(numpy_array);

  for (j = 0; j < num_rows; j++)

  {
    mf_sum = 0.0;
    for (i = 1; i < n_scalar_out; i++) {

      int index =
        j * n_scalar_out + i; // based on memory arrangement of input array
      int index2 =
        j * nspec_tot_recnet + i; // based on memory arrangement of output array

      mass_fractions[index - 1] = c_array[index2];

      if (mass_fractions[index - 1] < 0.0) {
        mass_fractions[index - 1] = 0.0;
      } // Zeroing negative values
      mf_sum = mf_sum + mass_fractions[index - 1];
    }

    mass_fractions[(j + 1) * n_scalar_out - 1] = 1.0 - mf_sum;

    // temperature[j] = c_array[j*nspec_tot_recnet];  // this can be commented
    // out when temperature is not updated
  }

  Py_DECREF(numpy_array);
  Py_DECREF(pValue);

  return 1;
}

static void
check_deeponet_vmap()

{

  int nrows = 100;
  double u0_star[nrows][n_scalar_out];
  double sie[nrows];
  double y_star, equiv;
  double temperature[nrows];

  clock_t start_t, end_t;
  double timetaken;
  int i, j, nstep;

  printf("The Call from UDF to C_DeepONet.\n");

  memset(u0_star, 0, nrows * n_scalar_out * sizeof(double));

  // Initial Condition
  for (i = 0; i < nrows; i++) {
    sie[i] = 1246205.3039284516 - 2E6;
    u0_star[i][0] = 1.23015463e-01;
    u0_star[i][1] = 3.59179936e-02;
    u0_star[i][2] = 1.45457045e-03;
    u0_star[i][3] = 5.54114394e-02;
    u0_star[i][4] = 2.48897239e-04;
    u0_star[i][5] = 5.17131877e-04;
    u0_star[i][6] = 3.58152378e-04;
    u0_star[i][7] = 8.22510570e-02;
    u0_star[i][8] = 2.72807060e-03;
    u0_star[i][9] = 0.0;
  }

  y_star = 1E-6; // In ms
  equiv = 1.0;

  // u0 =
  // np.array([1.1400000e+03, 6.0910187e-14, 5.6240503e-12, 5.5166412e-02, 1.3082714e-18,
  // 2.9821038e-14])

  start_t = clock();

  for (i = 0; i < 1; i++) {

    nstep =
      MF_DeepONet_VMAP(y_star, *u0_star, sie, temperature, nrows, n_scalar_out);
  }

  end_t = clock();

  printf(
    "Total time taken: %f\n", ((double)(end_t - start_t)) / CLOCKS_PER_SEC);

  // Print the returned values
  printf("Returned value 1: %f\n", u0_star[2][0]);
  printf("Returned value 2: %f\n", u0_star[2][1]);
  printf("Returned value 3: %f\n", u0_star[2][2]);
  printf("Returned value 4 (N2): %f\n", u0_star[2][n_scalar_out - 1]);
  printf("Returned Temperature: %f\n", temperature[2]);
  printf("Returned steps : %d\n", nstep);
}

#endif