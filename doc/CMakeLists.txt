FIND_PACKAGE(Doxygen QUIET)

MESSAGE("")
MESSAGE("================================ DOCUMENTATION =================================")
MESSAGE("Software that is required to build the documentation.")
MESSAGE("--------------------------------------------------------------------------------")
IF(DOXYGEN_FOUND)
	MESSAGE("[X] Doxygen")

	CONFIGURE_FILE(
		${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
		${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
	)
	CONFIGURE_FILE(
		${CMAKE_CURRENT_SOURCE_DIR}/header.html
		${CMAKE_CURRENT_BINARY_DIR}/header.html
		COPYONLY
	)
	CONFIGURE_FILE(
		${CMAKE_CURRENT_SOURCE_DIR}/stylesheet.css
		${CMAKE_CURRENT_BINARY_DIR}/stylesheet.css
		COPYONLY
	)
	CONFIGURE_FILE(
		${CMAKE_CURRENT_SOURCE_DIR}/mainpage.md
		${CMAKE_CURRENT_BINARY_DIR}/mainpage.md
		COPYONLY
	)
	CONFIGURE_FILE(
		${CMAKE_CURRENT_SOURCE_DIR}/figures/MainPageModel.png
		${CMAKE_CURRENT_BINARY_DIR}/figures/MainPageModel.png
		COPYONLY
	)
	CONFIGURE_FILE(
		${CMAKE_CURRENT_SOURCE_DIR}/InstallationInstructions.md
		${CMAKE_CURRENT_BINARY_DIR}/InstallationInstructions.md
		COPYONLY
	)
	CONFIGURE_FILE(
		${CMAKE_CURRENT_SOURCE_DIR}/Manual.md
		${CMAKE_CURRENT_BINARY_DIR}/Manual.md
		COPYONLY
	)
	CONFIGURE_FILE(
		${CMAKE_CURRENT_SOURCE_DIR}/Tutorials.md
		${CMAKE_CURRENT_BINARY_DIR}/Tutorials.md
		COPYONLY
	)

	ADD_CUSTOM_TARGET(
		documentation
		COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
		VERBATIM
	)

	SET(TBTK_DOCUMENTATION_ENABLED TRUE PARENT_SCOPE)
ELSE(DOXYGEN_FOUND)
	MESSAGE("[ ] Doxygen")
ENDIF(DOXYGEN_FOUND)
MESSAGE("================================================================================")
