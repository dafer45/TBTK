#ifndef COM_DAFER45_TBTK_MACRO
#define COM_DAFER45_TBTK_MACRO

#include <cstring>

/*#define TBTKAssert(expression, message)	\
	if(!(expression)){	\
		std::cout << "Error in file " << __FILE__ << ", line " << __LINE__ << ": " << message << "\n";	\
		exit(1);	\
	}*/

#define TBTKAssert(expression, function, message, hint)	\
	if(!(expression)){	\
		std::cout << "Error in " << function << "\n";	\
		std::cout << "\t" << message << "\n";	\
		if(std::strcmp(hint, "") != 0)	\
			std::cout << "\tHint: " << hint << "\n";	\
		std::cout << "\tWhere: " << __FILE__ << ", " << __LINE__ << "\n";	\
		exit(1);	\
	}

#endif
