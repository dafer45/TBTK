ADD_SUBDIRECTORY(ArnoldiIterator/TBTK/PropertyExtractor/)

INSTALL(
	DIRECTORY .
	DESTINATION include/TBTK/PropertyExtractor
	FILES_MATCHING PATTERN *.h
	PATTERN ArnoldiIterator* EXCLUDE
)
