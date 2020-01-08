#Avoid printing setting and printing information multiple times.
IF(NOT TBTK_VERSION_DEFINED)
	SET(TBTK_VERSION_MAJOR 2)
	SET(TBTK_VERSION_MINOR 0)
	SET(TBTK_VERSION_PATCH 0)
	SET(
		TBTK_VERSION
		${TBTK_VERSION_MAJOR}.${TBTK_VERSION_MINOR}.${TBTK_VERSION_PATCH}
	)

	ADD_DEFINITIONS(
		-DTBTK_VERSION_MAJOR=${TBTK_VERSION_MAJOR}
		-DTBTK_VERSION_MINOR=${TBTK_VERSION_MINOR}
		-DTBTK_VERSION_PATCH=${TBTK_VERSION_PATCH}
	)

	INCLUDE(TBTKGitInfo)

	MESSAGE("================================== ABOUT TBTK ==================================")
	MESSAGE("Version:\t${TBTK_VERSION}")
	MESSAGE("Git hash:\t${TBTK_VERSION_GIT_HASH}")
	MESSAGE("================================================================================")
	Message("")

	SET(TBTK_VERSION_DEFINED TRUE)
ENDIF(NOT TBTK_VERSION_DEFINED)
