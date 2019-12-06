FIND_PATH(
	PYTHON_INCLUDES
	NAMES Python.h
	PATH_SUFFIXES python2.7
)
FIND_LIBRARY(
	PYTHON_LIBRARIES
	NAMES python2.7
	PATH_SUFFIXES lib lib32 lib64
)

IF(PYTHON_INCLUDES AND PYTHON_LIBRARIES)
	SET(CMAKE_REQUIRED_INCLUDES ${PYTHON_INCLUDES})
	SET(CMAKE_REQUIRED_LIBRARIES ${PYTHON_LIBRARIES})
	INCLUDE(CheckCXXSourceRuns)
	UNSET(PYTHON_COMPILED CACHE)
	CHECK_CXX_SOURCE_RUNS(
		"
		#include <Python.h>
		#include <numpy/arrayobject.h>

#if PY_MAJOR_VERSION >= 3
		void* import_numpy(){
			import_array();
			return nullptr;
		}
#else
		void import_numpy(){
			import_array();
		}
#endif

		int main(int argc, char **argv){
			PyTuple_New(0);

			Py_SetProgramName(\"plotting\");
			Py_Initialize();
			import_numpy();

			PyObject *matplotlibname = PyString_FromString(\"matplotlib\");
			PyObject *matplotlib = PyImport_Import(matplotlibname);
			Py_DECREF(matplotlibname);
			if(!matplotlib)
				exit(1);

			PyObject *pyplotname = PyString_FromString(\"matplotlib.pyplot\");
			PyObject *pymod = PyImport_Import(pyplotname);
			if(!pymod)
				exit(1);

			return 0;
		}"
		PYTHON_COMPILED
	)
	IF(${PYTHON_COMPILED})
		INCLUDE(FindPackageHandleStandardArgs)
		FIND_PACKAGE_HANDLE_STANDARD_ARGS(
			Python FOUND_VAR Python_FOUND
			REQUIRED_VARS PYTHON_INCLUDES PYTHON_LIBRARIES
		)
	ENDIF(${PYTHON_COMPILED})
ENDIF(PYTHON_INCLUDES AND PYTHON_LIBRARIES)

#MARK_AS_ADVANCED(PYTHON_INCLDUES PYTHON_LIBRARIES)
