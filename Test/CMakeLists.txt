MESSAGE("")
set(TBTK_QUIET TRUE)
FIND_PACKAGE(TBTK QUIET)

FIND_PACKAGE(GTest QUIET)
MESSAGE("")
MESSAGE("=================================== TESTING ====================================")
MESSAGE("Libraries that are required to enable testing.")
MESSAGE("--------------------------------------------------------------------------------")
IF(GTest_FOUND)
	MESSAGE("[X] Google Test")
	INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS})

	FIND_PACKAGE(Threads REQUIRED)

	IF(TBTK_FOUND)
		MESSAGE("[X] TBTK (installed)")
		INCLUDE_DIRECTORIES(
			include/Core
			include/FiniteDifferences
			include/Utilities
		)

		FILE(GLOB_RECURSE SRC src/*.cpp)
		FOREACH(FILE ${SRC})
			GET_FILENAME_COMPONENT(FILENAME ${FILE} NAME)
			STRING(REPLACE ".cpp" "" FILENAME ${FILENAME})
			MESSAGE(${FILENAME})
			ADD_EXECUTABLE("TBTKTest${FILENAME}" ${FILE})
			ADD_TEST(NAME "${FILENAME}" COMMAND TBTKTest${FILENAME})

			TARGET_LINK_LIBRARIES("TBTKTest${FILENAME}" ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TBTK_LIBRARIES})
		ENDFOREACH(FILE ${SRC})

		SET(TBTK_TESTING_ENABLED TRUE PARENT_SCOPE)
	ELSE(TBTK_FOUND)
		MESSAGE("[ ] TBTK (installed)")
	ENDIF()
ELSE(GTest_FOUND)
	MESSAGE("[ ] Google Test")
ENDIF(GTest_FOUND)
MESSAGE("================================================================================")
