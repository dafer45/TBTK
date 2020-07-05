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

	FIND_PACKAGE(Threads QUIET REQUIRED)

	IF(TBTK_FOUND)
		MESSAGE("[X] TBTK (installed)")
		INCLUDE_DIRECTORIES(
			include/
			include/Core
			include/Elements
			include/Exceptions
			include/FiniteDifferences
			include/FourierTransform
			include/ManyParticle
			include/Uncategorized
			include/Utilities
			include/StatesAndOperators
		)

		FILE(GLOB_RECURSE SRC src/*.cpp)
		FOREACH(FILE ${SRC})
			#Convert soure filename to executable filename.
			STRING(FIND ${FILE} "Test" TEST_POSITION REVERSE)
			STRING(SUBSTRING ${FILE} ${TEST_POSITION} -1 FILENAME)
			STRING(REPLACE "/" "" FILENAME ${FILENAME})
			STRING(REPLACE "Test" "" FILENAME ${FILENAME})
			STRING(REPLACE ".cpp" "" FILENAME ${FILENAME})

			#Create executable and add to tests.
			ADD_EXECUTABLE(${FILENAME} ${FILE})
			ADD_TEST(NAME ${FILENAME} COMMAND ${FILENAME})

			TARGET_LINK_LIBRARIES(${FILENAME} ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TBTK_LIBRARIES})
		ENDFOREACH(FILE ${SRC})

		SET(TBTK_TESTING_ENABLED TRUE PARENT_SCOPE)
	ELSE(TBTK_FOUND)
		MESSAGE("[ ] TBTK (installed)")
	ENDIF()
ELSE(GTest_FOUND)
	MESSAGE("[ ] Google Test")
ENDIF(GTest_FOUND)
MESSAGE("================================================================================")
