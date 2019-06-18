CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(TBTK)

EXECUTE_PROCESS(
	COMMAND git rev-parse --short HEAD
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	OUTPUT_VARIABLE TBTK_VERSION_GIT_HASH
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

ENABLE_TESTING()

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/Modules/")

INCLUDE(TBTKVersionInfo)
INCLUDE(TBTKBuildOptions)

#Detect packages
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)
FIND_PACKAGE(BOOST QUIET)
FIND_PACKAGE(HDF5 QUIET)
FIND_PACKAGE(ARPACK QUIET)
FIND_PACKAGE(CUDA QUIET)
FIND_PACKAGE(CURL QUIET)
FIND_PACKAGE(Eigen3)
FIND_PACKAGE(FFTW3 QUIET)
FIND_PACKAGE(GMP QUIET)
FIND_PACKAGE(Gnuplot QUIET)
FIND_PACKAGE(OpenCV QUIET)
FIND_PACKAGE(OpenBLAS QUIET)
FIND_PACKAGE(OpenMP QUIET)
FIND_PACKAGE(SuperLU QUIET)
FIND_PACKAGE(wxWidgets QUIET)

#Detect packages with pkg-config
FIND_PACKAGE(PkgConfig QUIET)
IF(PKG_CONFIG_FOUND)
	PKG_CHECK_MODULES(LIBINT2 libint2)
ENDIF(PKG_CONFIG_FOUND)

#Print which extension enabling libraries that were found.
MESSAGE("")
MESSAGE("============================== OPTIONAL LIBRARIES ==============================")
MESSAGE("Libraries that enables extensions that were found (empty box means not found).")
MESSAGE("--------------------------------------------------------------------------------")
IF(ARPACK_FOUND)
	MESSAGE("[X] ARPACK")
ELSE(ARPACK_FOUND)
	MESSAGE("[ ] ARPACK")
ENDIF(ARPACK_FOUND)

IF(BOOST_FOUND)
	MESSAGE("[X] Boost")
ELSE(BOOST_FOUND)
	MESSAGE("[ ] Boost")
ENDIF(BOOST_FOUND)

IF(CUDA_FOUND)
	MESSAGE("[X] CUDA")
	ADD_DEFINITIONS(-DTBTK_CUDA_ENABLED)
ELSE(CUDA_FOUND)
	MESSAGE("[ ] CUDA")
ENDIF(CUDA_FOUND)

IF(CURL_FOUND)
	MESSAGE("[X] cURL")
ELSE(CURL_FOUND)
	MESSAGE("[ ] cURL")
ENDIF(CURL_FOUND)

IF(EIGEN3_FOUND)
	MESSAGE("[X] Eigen3")
	INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIRS})
ELSE(EIGEN3_FOUND)
	MESSAGE("[ ] Eigen3")
ENDIF(EIGEN3_FOUND)

IF(FFTW3_FOUND)
	MESSAGE("[X] FFTW3")
ELSE(FFTW3_FOUND)
	MESSAGE("[ ] FFTW3")
ENDIF(FFTW3_FOUND)

IF(GMP_FOUND)
	MESSAGE("[X] GMP")
	INCLUDE_DIRECTORIES(${GMP_INCLUDES})
ELSE(GMP_FOUND)
	MESSAGE("[ ] GMP")
ENDIF(GMP_FOUND)

IF(GNUPLOT_FOUND)
	MESSAGE("[X] Gnuplot")
ELSE(GNUPLOT_FOUND)
	MESSAGE("[ ] Gnuplot")
ENDIF(GNUPLOT_FOUND)

IF(HDF5_FOUND)
	MESSAGE("[X] HDF5")
ELSE(HDF5_FOUND)
	MESSAGE("[ ] HDF5")
ENDIF(HDF5_FOUND)

IF(LIBINT2_FOUND)
	MESSAGE("[X] Libint2")
	INCLUDE_DIRECTORIES(${LIBINT2_INCLUDE_DIRS})
	SET(TBTK_LINK_LIBRARIES ${LIBINT2_LIBRARIES})
ELSE(LIBINT2_FOUND)
	MESSAGE("[ ] Libint2")
ENDIF(LIBINT2_FOUND)

IF(OpenCV_FOUND)
	MESSAGE("[X] OpenCV")
ELSE(OpenCV_FOUND)
	MESSAGE("[ ] OpenCV")
ENDIF(OpenCV_FOUND)

IF(OpenBLAS_FOUND)
	MESSAGE("[X] OpenBLAS")
ELSE(OpenBLAS_FOUND)
	MESSAGE("[ ] OpenBLAS")
ENDIF(OpenBLAS_FOUND)

IF(OpenMP_FOUND OR OPENMP_FOUND)
	MESSAGE("[X] OpenMP")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	ADD_DEFINITIONS(-DTBTK_USE_OPEN_MP)
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
MESSAGE("================================================================================")

#Setup flags for building extensions and print which extensions that will be
#built.
MESSAGE("")
MESSAGE("================================== EXTENSIONS ==================================")
MESSAGE("Extensions that will be built (empty box means the extension will not be built).")
MESSAGE("--------------------------------------------------------------------------------")
IF(ARPACK_FOUND AND SuperLU_FOUND)
	MESSAGE("[X] ArnoldiIterator")
	SET(COMPILE_ARNOLDI_ITERATOR TRUE)
ELSE(ARPACK_FOUND AND SuperLU_FOUND)
	MESSAGE("[ ] ArnoldiIterator")
ENDIF(ARPACK_FOUND AND SuperLU_FOUND)

IF(CUDA_FOUND)
	MESSAGE("[X] CUDA")
	SET(COMPILE_CUDA TRUE)
	SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__")
ELSE(CUDA_FOUND)
	MESSAGE("[ ] CUDA")
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

IF(GMP_FOUND)
	MESSAGE("[X] ArbitraryPrecision")
	SET(COMPILE_ARBITRARY_PRECISION TRUE)
ELSE(GMP_FOUND)
	MESSAGE("[ ] ArbitraryPrecision")
ENDIF(GMP_FOUND)

IF(GNUPLOT_FOUND AND BOOST_FOUND)
	MESSAGE("[X] Plotter2")
	SET(COMPILE_PLOTTER2 TRUE)
ELSE(GNUPLOT_FOUND AND BOOST_FOUND)
	MESSAGE("[ ] Plotter2")
ENDIF(GNUPLOT_FOUND AND BOOST_FOUND)

IF(HDF5_FOUND)
	MESSAGE("[X] FileReader/FileWriter")
	SET(COMPILE_FILE_READER_WRITER TRUE)
	INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})
ELSE(HDF5_FOUND)
	MESSAGE("[ ] FileReader/FileWriter")
ENDIF(HDF5_FOUND)

IF(LIBINT2_FOUND AND EIGEN3_FOUND)
	MESSAGE("[X] Gaussian basis functions (libint2)")
	SET(COMPILE_LIBINT2 TRUE)
ELSE(LIBINT2_FOUND AND EIGEN3_FOUND)
	MESSAGE("[ ] Gaussian basis functions (libint2)")
ENDIF(LIBINT2_FOUND AND EIGEN3_FOUND)

IF(OpenCV_FOUND)
	MESSAGE("[X] Plotter")
	SET(COMPILE_PLOTTER TRUE)
	MESSAGE("[X] RayTracer")
	SET(COMPILE_RAY_TRACER TRUE)
ELSE(OpenCV_FOUND)
	MESSAGE("[ ] Plotter")
	MESSAGE("[ ] RayTracer")
ENDIF(OpenCV_FOUND)

IF(OpenBLAS_FOUND)
	MESSAGE("[X] OpenBLAS")
ELSE(OpenBLAS_FOUND)
	MESSAGE("[ ] OpenBLAS")
ENDIF(OpenBLAS_FOUND)

IF(SuperLU_FOUND)
	MESSAGE("[X] LinearEquationSolver")
	SET(COMPILE_LINEAR_EQUATION_SOLVER TRUE)
	MESSAGE("[X] LUSolver")
	SET(COMPILE_LU_SOLVER TRUE)
ELSE(SuperLU_FOUND)
	MESSAGE("[ ] LinearEquationSolver")
	MESSAGE("[ ] LUSolver")
ENDIF(SuperLU_FOUND)

IF(wxWidgets_FOUND AND OpenCV_FOUND)
	MESSAGE("[X] GUI")
	SET(COMPILE_GUI TRUE)
ELSE(wxWidgets_FOUND AND OpenCV_FOUND)
	MESSAGE("[ ] GUI")
ENDIF(wxWidgets_FOUND AND OpenCV_FOUND)

MESSAGE("================================================================================")

#C++ flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O3")

#Include paths
INCLUDE_DIRECTORIES(
	json/
	External/
#	hdf5/hdf5-build/hdf5/include/
#	${HDF5_INCLUDE_DIRS}
)

INCLUDE_DIRECTORIES(
	Lib/include/
	Lib/include/Builders/
	Lib/include/Core/
	Lib/include/Elements/
	Lib/include/Exceptions/
	Lib/include/FiniteDifferences/
	Lib/include/GUI/
#	Lib/include/Lattices/
#	Lib/include/Lattices/D2/
#	Lib/include/Lattices/D3/
	Lib/include/ManyParticle/
	Lib/include/ManyParticle/TBTK/FockStateMap/
	Lib/include/ManyParticle/TBTK/FockStateRule/
#	Lib/include/Properties/
#	Lib/include/PropertyExtractors/
#	Lib/include/Solvers/
	Lib/include/Resource/
	Lib/include/SpecializedSolvers/
	Lib/include/StatesAndOperators/
	Lib/include/Uncategorized/
	Lib/include/Utilities/
)

IF(${COMPILE_ARNOLDI_ITERATOR})
#	INCLUDE_DIRECTORIES(Lib/include/TBTK/Solver/ArnoldiIterator/)
#	INCLUDE_DIRECTORIES(Lib/include/TBTK/PropertyExtractor/ArnoldiIterator/)
ENDIF(${COMPILE_ARNOLDI_ITERATOR})

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

IF(${COMPILE_PLOTTER2})
	INCLUDE_DIRECTORIES(Lib/include/Plotter2/)
ENDIF(${COMPILE_PLOTTER2})

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
ADD_SUBDIRECTORY(External/TBTK/External/Boost)
ADD_SUBDIRECTORY(Templates/)
ADD_SUBDIRECTORY(Tools/)
ADD_SUBDIRECTORY(Visualization/python/)
ADD_SUBDIRECTORY(Test/)
ADD_SUBDIRECTORY(doc/)

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
	CMake/TBTKConfigTemplate.cmake
	"${CMAKE_CURRENT_BINARY_DIR}/CMake/TBTKConfig.cmake"
	COPYONLY
)

#Copy Find modules to build folder.
CONFIGURE_FILE(
	CMake/Modules/FindARPACK.cmake
	"${CMAKE_CURRENT_BINARY_DIR}/CMake/Modules/FindARPACK.cmake"
	COPYONLY
)
CONFIGURE_FILE(
	CMake/Modules/FindFFTW3.cmake
	"${CMAKE_CURRENT_BINARY_DIR}/CMake/Modules/FindFFTW3.cmake"
	COPYONLY
)
CONFIGURE_FILE(
	CMake/Modules/FindGMP.cmake
	"${CMAKE_CURRENT_BINARY_DIR}/CMake/Modules/FindGMP.cmake"
	COPYONLY
)
CONFIGURE_FILE(
	CMake/Modules/FindOpenBLAS.cmake
	"${CMAKE_CURRENT_BINARY_DIR}/CMake/Modules/FindOpenBLAS.cmake"
	COPYONLY
)
CONFIGURE_FILE(
	CMake/Modules/FindSuperLU.cmake
	"${CMAKE_CURRENT_BINARY_DIR}/CMake/Modules/FindSuperLU.cmake"
	COPYONLY
)

#Copy TBTKVersionInfo.cmake, TBTKGitInfo.cmake, and TBTKBuildOptions to build
#folder.
CONFIGURE_FILE(
	CMake/Modules/TBTKVersionInfo.cmake
	"${CMAKE_CURRENT_BINARY_DIR}/CMake/Modules/TBTKVersionInfo.cmake"
	COPYONLY
)
CONFIGURE_FILE(
	CMake/Modules/TBTKGitInfo.cmake
	${CMAKE_CURRENT_BINARY_DIR}/CMake/Modules/TBTKGitInfo.cmake
)
CONFIGURE_FILE(
	CMake/Modules/TBTKBuildOptions.cmake
	${CMAKE_CURRENT_BINARY_DIR}/CMake/Modules/TBTKBuildOptions.cmake
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
	FILES ${CMAKE_CURRENT_BINARY_DIR}/CMake/TBTKConfig.cmake "${CMAKE_CURRENT_BINARY_DIR}/CMake/TBTKVersion.cmake"
	DESTINATION ${ConfigPackageLocation}
	COMPONENT Devel
)
INSTALL(
	DIRECTORY ${TBTK_BINARY_DIR}/CMake/Modules
	DESTINATION ${ConfigPackageLocation}
	FILES_MATCHING PATTERN *.cmake
)

#MESSAGE("Installation path that will be used if installed: ${CMAKE_INSTALL_PREFIX}")

IF(UNIX)
	MESSAGE("")
	MESSAGE("================================= WHAT'S NEXT? =================================")
	MESSAGE("TBTK build scripts successfully generated. The following options are now")
	MESSAGE("available.")
	MESSAGE("--------------------------------------------------------------------------------")
	MESSAGE("make\t\t\tBuild TBTK.")
	MESSAGE("sudo make install\tInstall TBTK to '${CMAKE_INSTALL_PREFIX}'.")
#	MESSAGE("\nType 'make' to build TBTK, followed by 'sudo make install' to install.")
	IF(${TBTK_TESTING_ENABLED})
		MESSAGE("make test\t\tRun tests.")
#		MESSAGE("Type 'make test' to run tests.")
	ENDIF(${TBTK_TESTING_ENABLED})
	IF(${TBTK_DOCUMENTATION_ENABLED})
		MESSAGE("make documentation\tBuild the documentation. The documentation will be found")
		MESSAGE("\t\t\tat '${CMAKE_BINARY_DIR}/doc/html/index.html'.")
#		MESSAGE("Type 'make documentation' to run tests.")
	ENDIF(${TBTK_DOCUMENTATION_ENABLED})
	MESSAGE("--------------------------------------------------------------------------------")
	MESSAGE("Once TBTK has been installed, you can also type 'TBTK' in the terminal for")
	MESSAGE("further instructions.")
	MESSAGE("================================================================================\n")
ENDIF(UNIX)
