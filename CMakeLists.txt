CMAKE_MINIMUM_REQUIRED(VERSION 3.5.1)

PROJECT(TBTK)

SET(PROJECT_MAJOR_VERSION 0)
SET(PROJECT_MINOR_VERSION 9)
SET(PROJECT_PATCH_VERSION 5)
SET(
	PROJECT_VERSION
	${PROJECT_MAJOR_VERSION}.${PROJECT_MINOR_VERSION}.${PROJECT_PATCH_VERSION}
)

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/Modules/")

#Detect packages
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)
FIND_PACKAGE(HDF5 REQUIRED)
FIND_PACKAGE(ARPACK QUIET)
FIND_PACKAGE(CUDA QUIET)
FIND_PACKAGE(CURL QUIET)
FIND_PACKAGE(FFTW3 QUIET)
FIND_PACKAGE(OpenCV QUIET)
FIND_PACKAGE(OpenMP QUIET)
FIND_PACKAGE(SuperLU QUIET)
FIND_PACKAGE(wxWidgets QUIET)

MESSAGE(${HDF5_LIBRARY_DIRS})

#Print which extention enabling libraries that were found.
MESSAGE("\nLibraries that enables extentions that were found (empty box means not found).")
IF(ARPACK_FOUND)
	MESSAGE("[X] ARPACK")
ELSE(ARPACK_FOUND)
	MESSAGE("[ ] ARPACK")
ENDIF(ARPACK_FOUND)

IF(CUDA_FOUND)
	MESSAGE("[X] Cuda")
ELSE(CUDA_FOUND)
	MESSAGE("[ ] Cuda")
ENDIF(CUDA_FOUND)

IF(CURL_FOUND)
	MESSAGE("[X] cURL")
ELSE(CURL_FOUND)
	MESSAGE("[ ] cURL")
ENDIF(CURL_FOUND)

IF(FFTW3_FOUND)
	MESSAGE("[X] FFTW3")
ELSE(FFTW3_FOUND)
	MESSAGE("[ ] FFTW3")
ENDIF(FFTW3_FOUND)

IF(OpenCV_FOUND)
	MESSAGE("[X] OpenCV")
ELSE(OpenCV_FOUND)
	MESSAGE("[ ] OpenCV")
ENDIF(OpenCV_FOUND)

IF(OpenMP_FOUND)
	MESSAGE("[X] OpenMP")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ELSE(OpenMP_FOUND)
	MESSAGE("[X] OpenMP")
ENDIF(OpenMP_FOUND)

IF(SuperLU_FOUND)
	MESSAGE("[X] SuperLU")
ELSE(SuperLU_FOUND)
	MESSAGE("[ ] SuperLU")
ENDIF(SuperLU_FOUND)

IF(wxWidgets_FOUND)
	MESSAGE("[X] wxWidgets")
#	INCLUDE_DIRECTORIES(${wxWidgets_INCLUDE_DIRS})
	INCLUDE(${wxWidgets_USE_FILE})
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${wxWidgets_CXX_FLAGS}")
	SET(TBTK_LINK_LIBRARIES ${wxWidgets_LIBRARIES})
ELSE(wxWidgets_FOUND)
	MESSAGE("[ ] wxWidgets")
ENDIF(wxWidgets_FOUND)

#Setup flags for building extentions and print which extentions that will be
#built.
MESSAGE("\nExtenstions that will be built (empty box means the extention will not be built).")
IF(ARPACK_FOUND AND SuperLU_FOUND)
	MESSAGE("[X] ArnoldiSolver")
	SET(COMPILE_ARNOLDI_SOLVER TRUE)
ELSE(ARPACK_FOUND AND SuperLU_FOUND)
	MESSAGE("[ ] ArnoldiSolver")
ENDIF(ARPACK_FOUND AND SuperLU_FOUND)

IF(CUDA_FOUND)
	MESSAGE("[X] Cuda")
	SET(COMPILE_CUDA TRUE)
	SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES")
ELSE(CUDA_FOUND)
	MESSAGE("[ ] Cuda")
ENDIF(CUDA_FOUND)

IF(CURL_FOUND)
	MESSAGE("[X] Resource")
	SET(COMPILE_RESOURCE TRUE)
	MESSAGE("[X] DataManager")
	SET(COMPILE_DATA_MANAGER TRUE)
ELSE(CURL_FOUND)
	MESSAGE("[ ] Resource")
	MESSAGE("[ ] DataManager")
ENDIF(CURL_FOUND)

IF(FFTW3_FOUND)
	MESSAGE("[X] FourierTransform")
	SET(COMPILE_FOURIER_TRANSFORM TRUE)
ELSE(FFTW3_FOUND)
	MESSAGE("[ ] FourierTransform")
ENDIF(FFTW3_FOUND)

IF(OpenCV_FOUND)
	MESSAGE("[X] Plotter")
	SET(COMPILE_PLOTTER TRUE)
	MESSAGE("[X] RayTracer")
	SET(COMPILE_RAY_TRACER TRUE)
ELSE(OpenCV_FOUND)
	MESSAGE("[ ] Plotter")
	MESSAGE("[ ] RayTracer")
ENDIF(OpenCV_FOUND)

IF(SuperLU_FOUND)
	MESSAGE("[X] LinearEquationSolver")
	SET(COMPILE_LINEAR_EQUATION_SOLVER TRUE)
	MESSAGE("[X] LUSolver")
	SET(COMPILE_LU_SOLVER TRUE)
ELSE(SuperLU_FOUND)
	MESSAGE("[ ] LinearEquationSolver")
	MESSAGE("[ ] LUSolver")
ENDIF(SuperLU_FOUND)

IF(wxWidgets_FOUND)
	MESSAGE("[X] GUI")
	SET(COMPILE_GUI TRUE)
ELSE(wxWidgets_FOUND)
	MESSAGE("[ ] GUI")
ENDIF(wxWidgets_FOUND)
MESSAGE("\n")

#C++ flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O3")

#Include paths
INCLUDE_DIRECTORIES(
	json/
#	hdf5/hdf5-build/hdf5/include/
	${HDF5_INCLUDE_DIRS}
)

INCLUDE_DIRECTORIES(
	Lib/include/Builders/
	Lib/include/Core/
	Lib/include/Elements/
	Lib/include/Exceptions/
	Lib/include/GUI/
	Lib/include/Lattices/
	Lib/include/Lattices/D2/
	Lib/include/Lattices/D3/
	Lib/include/ManyBody/
	Lib/include/ManyBody/FockStateMap/
	Lib/include/ManyBody/FockStateRule/
	Lib/include/Properties/
	Lib/include/PropertyExtractors/
	Lib/include/Solvers/
	Lib/include/SpecializedSolvers/
	Lib/include/StatesAndOperators/
	Lib/include/Uncategorized/
	Lib/include/Utilities/
)

IF(${COMPILE_ARNOLDI_SOLVER})
	INCLUDE_DIRECTORIES(LIB/include/Solvers/ArnoldiSolver/)
ENDIF(${COMPILE_ARNOLDI_SOLVER})

IF(${COMPILE_DATA_MANAGER})
	INCLUDE_DIRECTORIES(Lib/include/Utilities/DataManager/)
ENDIF(${COMPILE_DATA_MANAGER})

IF(${COMPILE_FOURIER_TRANSFORM})
	INCLUDE_DIRECTORIES(LIB/include/Utilities/FourierTransform)
ENDIF(${COMPILE_FOURIER_TRANSFORM})

IF(${COMPILE_GUI})
	INCLUDE_DIRECTORIES(LIB/include/GUI/)
ENDIF(${COMPILE_GUI})

IF(${COMPILE_LINEAR_EQUATION_SOLVER})
	INCLUDE_DIRECTORIES(LIB/include/Solvers/LinearEquationSolver/)
ENDIF(${COMPILE_LINEAR_EQUATION_SOLVER})

IF(${COMPILE_PLOTTER})
	INCLUDE_DIRECTORIES(LIB/include/Utilities/Plotter/)
ENDIF(${COMPILE_PLOTTER})

IF(${COMPILE_RESOURCE})
	INCLUDE_DIRECTORIES(LIB/include/Utilities/Resource/)
ENDIF(${COMPILE_RESOURCE})

IF(${COMPILE_RAY_TRACER})
	INCLUDE_DIRECTORIES(LIB/include/Utilities/RayTracer/)
ENDIF(${COMPILE_RAY_TRACER})

#Add subdirectory
ADD_SUBDIRECTORY(Lib/include/)
ADD_SUBDIRECTORY(Lib/src/)
ADD_SUBDIRECTORY(json/)

#INSTALL(
#	TARGETS TBTK
#	LIBRARY DESTINATION lib
#)