#ifndef COM_DAFER45_TBTK_INDEX_EXCEPTION
#define COM_DAFER45_TBTK_INDEX_EXCEPTION

#include "TBTK/Exception.h"

#include <string>

namespace TBTK{

class IndexException : public Exception{
public:
	/** Constructor. */
	IndexException(
		const std::string& function,
		const std::string& where,
		const std::string& message,
		const std::string& hint
	);

	/** Destructor. */
	virtual ~IndexException();
private:
};

};	//End of namespace TBTK

#endif
