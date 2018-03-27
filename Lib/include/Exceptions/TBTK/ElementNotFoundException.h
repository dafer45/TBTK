#ifndef COM_DAFER45_TBTK_ELEMENT_NOT_FOUND_EXCEPTION
#define COM_DAFER45_TBTK_ELEMENT_NOT_FOUND_EXCEPTION

#include "TBTK/Exception.h"

#include <string>

namespace TBTK{

class ElementNotFoundException : public Exception{
public:
	/** Constructor. */
	ElementNotFoundException(
		const std::string& function,
		const std::string& where,
		const std::string& message,
		const std::string& hint
	);

	/** Destructor. */
	virtual ~ElementNotFoundException();
private:
};

};	//End of namespace TBTK

#endif
