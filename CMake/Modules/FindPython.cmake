UNSET(PYTHON_INCLUDES CACHE)
UNSET(PYTHON_LIBRARIES CACHE)
FIND_PATH(
	PYTHON_INCLUDES
	NAMES Python.h
	PATH_SUFFIXES python3.0 python3.1 python3.2 python3.3 python3.4 python3.5 python3.6 python3.7 python3.8 python3.9 python3.10 python3.0m python3.1m python3.2m python3.3m python3.4m python3.5m python3.6m python3.7m python3.8m python3.9m python3.0d python3.1d python3.2d python3.3d python3.4d python3.5d python3.6d python3.7d python3.8d python3.9d python3.0u python3.1u python3.2u python3.3u python3.4u python3.5u python3.6u python3.7u python3.8u python3.9u
)
FIND_LIBRARY(
	PYTHON_LIBRARIES
	NAMES python3.0 python3.1 python3.2 python3.3 python3.4 python3.5 python3.6 python3.7 python3.8 python3.9 python3.10 python3.0m python3.1m python3.2m python3.3m python3.4m python3.5m python3.6m python3.7m python3.8m python3.9m python3.0d python3.1d python3.2d python3.3d python3.4d python3.5d python3.6d python3.7d python3.8d python3.9d python3.0u python3.1u python3.2u python3.3u python3.4u python3.5u python3.6u python3.7u python3.8u python3.9u
	PATH_SUFFIXES lib lib32 lib64
)
IF(("${PYTHON_INCLUDES}" MATCHES "PYTHON_INCLUDES-NOTFOUND") OR ("${PYTHON_LIBRARIES}" MATCHES "PYTHON_LIBRARIES-NOTFOUND"))
	UNSET(PYTHON_INCLUDES CACHE)
	UNSET(PYTHON_LIBRARIES CACHE)
ENDIF(("${PYTHON_INCLUDES}" MATCHES "PYTHON_INCLUDES-NOTFOUND") OR ("${PYTHON_LIBRARIES}" MATCHES "PYTHON_LIBRARIES-NOTFOUND"))
IF((NOT DEFINED PYTHON_INCLUDES) OR (NOT DEFINED PYTHON_LIBRARIES))
	UNSET(PYTHON_INCLUDES CACHE)
	UNSET(PYTHON_LIBRARIES CACHE)
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
		SET(TBTK_PYTHON_VERSION 2)
	ENDIF(PYTHON_INCLUDES AND PYTHON_LIBRARIES)
ELSE((NOT DEFINED PYTHON_INCLUDES) OR (NOT DEFINED PYTHON_LIBRARIES))
	SET(TBTK_PYTHON_VERSION 3)
ENDIF((NOT DEFINED PYTHON_INCLUDES) OR (NOT DEFINED PYTHON_LIBRARIES))

IF(PYTHON_INCLUDES AND PYTHON_LIBRARIES)
	SET(CMAKE_REQUIRED_INCLUDES ${PYTHON_INCLUDES})
	SET(CMAKE_REQUIRED_LIBRARIES ${PYTHON_LIBRARIES})
	INCLUDE(CheckCXXSourceRuns)
	UNSET(TBTK_PYTHON_COMPILED CACHE)
	CHECK_CXX_SOURCE_RUNS(
		"
		#include <Python.h>
		int main(int argc, char **argv){
			return 0;
		}
		"
		TBTK_PYTHON_COMPILED
	)
	UNSET(TBTK_PYTHON_NUMPY_COMPILED CACHE)
	CHECK_CXX_SOURCE_RUNS(
		"
		#include <Python.h>
		#include <numpy/arrayobject.h>
		int main(int argc, char **argv){
			return 0;
		}
		"
		TBTK_PYTHON_NUMPY_COMPILED
	)
	UNSET(TBTK_PYTHON_MATPLOTLIB_COMPILED CACHE)
	CHECK_CXX_SOURCE_RUNS(
		"
		#include <Python.h>
		#include <numpy/arrayobject.h>
#if PY_MAJOR_VERSION >= 3
#		define PyString_FromString PyUnicode_FromString
#endif
		int main(int argc, char **argv){
#if PY_MAJOR_VERSION >= 3
			wchar_t name[] = L\"plotting\";
#else
			char name[] = \"plotting\";
#endif
			Py_SetProgramName(name);
			Py_Initialize();
			PyObject *matplotlibName = PyString_FromString(\"matplotlib\");
			PyObject *matplotlib = PyImport_Import(matplotlibName);
			Py_DECREF(matplotlibName);
			if(!matplotlib)
				exit(1);
			return 0;
		}
		"
		TBTK_PYTHON_MATPLOTLIB_COMPILED
	)
	UNSET(PYTHON_COMPILED CACHE)
	CHECK_CXX_SOURCE_RUNS(
		"
		#include <Python.h>
		#include <numpy/arrayobject.h>

#if PY_MAJOR_VERSION >= 3
#		define PyString_FromString PyUnicode_FromString

		void* import_numpy(){
			import_array();
			return NULL;
		}
#else
		void import_numpy(){
			import_array();
		}
#endif

		int main(int argc, char **argv){
#if PY_MAJOR_VERSION >= 3
			wchar_t name[] = L\"plotting\";
#else
			char name[] = \"plotting\";
#endif
			Py_SetProgramName(name);
			Py_Initialize();
			import_numpy();

			PyObject *matplotlibName = PyString_FromString(\"matplotlib\");
			PyObject *matplotlib = PyImport_Import(matplotlibName);
			Py_DECREF(matplotlibName);
			if(!matplotlib)
				exit(1);

			PyObject_CallMethod(
				matplotlib,
				const_cast<char*>(\"use\"),
				const_cast<char*>(\"s\"),
				\"Agg\"
			);

			PyObject *pyplotName = PyString_FromString(\"matplotlib.pyplot\");
			PyObject *pyplot = PyImport_Import(pyplotName);
			if(!pyplot)
				exit(1);

			PyObject *emptyTuple = PyTuple_New(0);

			PyObject *clf = PyObject_GetAttrString(pyplot, \"clf\");
			PyObject *clfReturnValue = PyObject_CallObject(
				clf,
				emptyTuple
			);
			if(!clfReturnValue)
				exit(1);
			Py_DECREF(clfReturnValue);

			Py_Finalize();

			return 0;
		}"
		PYTHON_COMPILED
	)
	IF(${PYTHON_COMPILED})
		UNSET(TBTK_MATPLOTLIB_DO_NOT_FORCE_AGG CACHE)
		CHECK_CXX_SOURCE_RUNS(
			"
			#include <Python.h>
			#include <numpy/arrayobject.h>

#if PY_MAJOR_VERSION >= 3
#			define PyString_FromString PyUnicode_FromString

			void* import_numpy(){
				import_array();
				return NULL;
			}
#else
			void import_numpy(){
				import_array();
			}
#endif

			int main(int argc, char **argv){
#if PY_MAJOR_VERSION >= 3
				wchar_t name[] = L\"plotting\";
#else
				char name[] = \"plotting\";
#endif
				Py_SetProgramName(name);
				Py_Initialize();
				import_numpy();

				PyObject *matplotlibName = PyString_FromString(\"matplotlib\");
				PyObject *matplotlib = PyImport_Import(matplotlibName);
				Py_DECREF(matplotlibName);
				if(!matplotlib)
					exit(1);

				PyObject *pyplotName = PyString_FromString(\"matplotlib.pyplot\");
				PyObject *pyplot = PyImport_Import(pyplotName);
				if(!pyplot)
					exit(1);

				PyObject *emptyTuple = PyTuple_New(0);

				PyObject *clf = PyObject_GetAttrString(pyplot, \"clf\");
				PyObject *clfReturnValue = PyObject_CallObject(
					clf,
					emptyTuple
				);
				if(!clfReturnValue)
					exit(1);
				Py_DECREF(clfReturnValue);

				Py_Finalize();

				return 0;
			}"
			TBTK_MATPLOTLIB_DO_NOT_FORCE_AGG
		)
		IF(TBTK_MATPLOTLIB_DO_NOT_FORCE_AGG)
			ADD_DEFINITIONS(-DTBTK_MATPLOTLIB_DO_NOT_FORCE_AGG)
		ENDIF(TBTK_MATPLOTLIB_DO_NOT_FORCE_AGG)

		INCLUDE(FindPackageHandleStandardArgs)
		FIND_PACKAGE_HANDLE_STANDARD_ARGS(
			Python FOUND_VAR Python_FOUND
			REQUIRED_VARS PYTHON_INCLUDES PYTHON_LIBRARIES
		)
	ENDIF(${PYTHON_COMPILED})
ENDIF(PYTHON_INCLUDES AND PYTHON_LIBRARIES)

#MARK_AS_ADVANCED(PYTHON_INCLDUES PYTHON_LIBRARIES)
