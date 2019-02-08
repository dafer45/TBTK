#Avoid setting and printing information multiple times.
IF(NOT TBTK_BUILD_OPTIONS_DEFINED)
	#Build options.
	SET(TBTK_WRAP_PRIMITIVE_TYPES TRUE)

	#Consequences.
	IF(TBTK_WRAP_PRIMITIVE_TYPES)
		ADD_DEFINITIONS(-DTBTK_WRAP_PRIMITIVE_TYPES=1)
	ELSE(TBTK_WRAP_PRIMITIVE_TYPES)
		ADD_DEFINITIONS(-DTBTK_WRAP_PRIMITIVE_TYPES=0)
	ENDIF(TBTK_WRAP_PRIMITIVE_TYPES)

	#Print status.
	MESSAGE("================================ Build options =================================")
	MESSAGE("Wrap primitive types:\t${TBTK_WRAP_PRIMITIVE_TYPES}")
	MESSAGE("================================================================================")
	Message("")

	SET(TBTK_BUILD_OPTIONS_DEFINED TRUE)
ENDIF(NOT TBTK_BUILD_OPTIONS_DEFINED)
