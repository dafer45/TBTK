CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12.2)

PROJECT(TBTKTest)

#Detect packages
FIND_PACKAGE(GTest REQUIRED)
INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS})

FIND_PACKAGE(Threads REQUIRED)

FIND_PACKAGE(TBTK REQUIRED)
INCLUDE_DIRECTORIES(
	include/Core
)

#LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O3")

ENABLE_TESTING()

FILE(GLOB SRC src/*)
ADD_EXECUTABLE(TBTKTest ${SRC})

TARGET_LINK_LIBRARIES(TBTKTest ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TBTK_LIBRARIES})