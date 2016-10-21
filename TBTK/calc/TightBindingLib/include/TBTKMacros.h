#ifndef COM_DAFER45_TBTK_MACRO
#define COM_DAFER45_TBTK_MACRO

#include "Streams.h"

#include <cstring>

#define TBTKAssert(expression, function, message, hint)	\
	if(!(expression)){	\
		TBTK::Util::Streams::err << "Error in " << function << "\n";	\
		TBTK::Util::Streams::err << "\t" << message << "\n";	\
		if(std::strcmp(hint, "") != 0)	\
			TBTK::Util::Streams::err << "\tHint: " << hint << "\n";	\
		TBTK::Util::Streams::err << "\tWhere: " << __FILE__ << ", " << __LINE__ << "\n";	\
		Util::Streams::closeLog();	\
		exit(1);	\
	}

#endif
