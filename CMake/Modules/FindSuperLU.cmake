FIND_PATH(
	SUPER_LU_INCLUDES
	NAMES supermatrix.h
	PATHS ${SUPER_LU_INCLUDE_PATH}
	PATH_SUFFIXES SRC superlu
)
FIND_LIBRARY(
	SUPER_LU_LIBRARIES
	NAMES superlu
	PATHS "${SUPER_LU_LIBRARY_PATH}"
	PATH_SUFFIXES lib lib32 lib64 SRC
)

SET(CMAKE_REQUIRED_INCLUDES ${SUPER_LU_INCLUDES})
SET(CMAKE_REQUIRED_LIBRARIES ${SUPER_LU_LIBRARIES} lapack blas)
INCLUDE(CheckCXXSourceCompiles)
CHECK_CXX_SOURCE_COMPILES(
	"
	#include <slu_ddefs.h>
	#include <slu_zdefs.h>

	int main(int argc, char **argv){
		superlu_options_t options;
		int *etree;
		SuperMatrix matrixCP;
		int panelSize;
		int relax;
		int lwork;
		GlobalLU_t glu;
		int info;

		SuperMatrix *L;
		SuperMatrix *U;
		int *rowPermutations;
		int *columnPermutations;
		SuperLUStat_t *statistics;

		zgstrf(
                        &options,
                        &matrixCP,
                        relax,
                        panelSize,
                        etree,
                        NULL,
                        lwork,
                        columnPermutations,
                        rowPermutations,
                        L,
                        U,
                        &glu,
                        statistics,
                        &info
                );

		return 0;
	}"
	SUPER_LU_COMPILED
)

IF(${SUPER_LU_COMPILED})
	INCLUDE(FindPackageHandleStandardArgs)
	FIND_PACKAGE_HANDLE_STANDARD_ARGS(
		SuperLU FOUND_VAR SuperLU_FOUND
		REQUIRED_VARS SUPER_LU_INCLUDES SUPER_LU_LIBRARIES
	)
ENDIF(${SUPER_LU_COMPILED})

#MARK_AS_ADVANCED(SUPER_LU_INCLDUES SUPER_LU_LIBRARIES)
