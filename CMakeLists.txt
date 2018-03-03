CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12.2)

PROJECT(TBTK)

SET(TBTK_MAJOR_VERSION 0)
SET(TBTK_MINOR_VERSION 9)
SET(TBTK_PATCH_VERSION 5)
SET(
	TBTK_VERSION
	${TBTK_MAJOR_VERSION}.${TBTK_MINOR_VERSION}.${TBTK_PATCH_VERSION}
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

IF(OpenMP_FOUND OR OPENMP_FOUND)
	MESSAGE("[X] OpenMP")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ELSE(OpenMP_FOUND OR OPENMP_FOUND)
	MESSAGE("[ ] OpenMP")
ENDIF(OpenMP_FOUND OR OPENMP_FOUND)

IF(SuperLU_FOUND)
	MESSAGE("[X] SuperLU")
	INCLUDE_DIRECTORIES(${SUPER_LU_INCLUDES})
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
	Lib/include/
	Lib/include/Builders/
	Lib/include/Core/
	Lib/include/Elements/
	Lib/include/Exceptions/
	Lib/include/GUI/
	Lib/include/Lattices/
#	Lib/include/Lattices/D2/
#	Lib/include/Lattices/D3/
	Lib/include/ManyBody/
	Lib/include/ManyBody/TBTK/FockStateMap/
	Lib/include/ManyBody/TBTK/FockStateRule/
#	Lib/include/Properties/
#	Lib/include/PropertyExtractors/
#	Lib/include/Solvers/
	Lib/include/Resource/
	Lib/include/SpecializedSolvers/
	Lib/include/StatesAndOperators/
	Lib/include/Uncategorized/
	Lib/include/Utilities/
)

IF(${COMPILE_ARNOLDI_SOLVER})
	INCLUDE_DIRECTORIES(Lib/include/TBTK/Solver/ArnoldiIterator/)
	INCLUDE_DIRECTORIES(Lib/include/TBTK/PropertyExtractor/ArnoldiIterator/)
ENDIF(${COMPILE_ARNOLDI_SOLVER})

IF(${COMPILE_DATA_MANAGER})
	INCLUDE_DIRECTORIES(Lib/include/DataManager/)
ENDIF(${COMPILE_DATA_MANAGER})

IF(${COMPILE_FOURIER_TRANSFORM})
	INCLUDE_DIRECTORIES(Lib/include/FourierTransform/)
ENDIF(${COMPILE_FOURIER_TRANSFORM})

IF(${COMPILE_GUI})
	INCLUDE_DIRECTORIES(Lib/include/GUI/)
ENDIF(${COMPILE_GUI})

IF(${COMPILE_LINEAR_EQUATION_SOLVER})
	INCLUDE_DIRECTORIES(Lib/include/TBTK/Solver/LinearEquationSolver/)
ENDIF(${COMPILE_LINEAR_EQUATION_SOLVER})

IF(${COMPILE_LU_SOLVER})
	INCLUDE_DIRECTORIES(Lib/include/TBTK/Solver/LUSolver/)
ENDIF(${COMPILE_LU_SOLVER})

IF(${COMPILE_PLOTTER})
	INCLUDE_DIRECTORIES(Lib/include/Plotter/)
ENDIF(${COMPILE_PLOTTER})

IF(${COMPILE_RESOURCE})
	INCLUDE_DIRECTORIES(Lib/include/Resource/)
ENDIF(${COMPILE_RESOURCE})

IF(${COMPILE_RAY_TRACER})
	INCLUDE_DIRECTORIES(Lib/include/RayTracer/)
ENDIF(${COMPILE_RAY_TRACER})

#Add subdirectory.
ADD_SUBDIRECTORY(Lib/include/)
ADD_SUBDIRECTORY(Lib/src/)
ADD_SUBDIRECTORY(json/TBTK)

#Create TBTKVersion.cmake.
INCLUDE(CMakePackageConfigHelpers)
WRITE_BASIC_PACKAGE_VERSION_FILE(
	"${CMAKE_CURRENT_BINARY_DIR}/CMake/TBTKVersion.cmake"
	VERSION ${TBTK_VERSION}
	COMPATIBILITY AnyNewerVersion
)

#Create TBTKTargets.cmake.
EXPORT(
	EXPORT TBTKLibraries
	FILE "${CMAKE_CURRENT_BINARY_DIR}/CMake/TBTKTargets.cmake"
	NAMESPACE TBTK::
)

#Copy TBTKConfigure.cmake to installation folder.
CONFIGURE_FILE(
	CMake/TBTKConfig.cmake
	"${CMAKE_CURRENT_BINARY_DIR}/CMake/TBTK/TBTKConfig.cmake"
	COPYONLY
)

#Installation rules for configuration files.
SET(ConfigPackageLocation lib/TBTK/CMake)
INSTALL(
	EXPORT TBTKLibraries
	FILE TBTKTargets.cmake
	NAMESPACE TBTK::
	DESTINATION ${ConfigPackageLocation}
)
INSTALL(
	FILES CMake/TBTKConfig.cmake "${CMAKE_CURRENT_BINARY_DIR}/CMake/TBTKVersion.cmake"
	DESTINATION ${ConfigPackageLocation}
	COMPONENT Devel
)
INSTALL(
	DIRECTORY ./CMake/Modules
	DESTINATION ${ConfigPackageLocation}
	FILES_MATCHING PATTERN *.cmake
)
INSTALL(
	FILES CMake/TBTKConfig.cmake
	DESTINATION ${ConfigPackageLocation}
)
