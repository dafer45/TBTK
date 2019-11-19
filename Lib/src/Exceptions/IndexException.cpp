#include "TBTK/IndexException.h"

namespace TBTK{

IndexException::IndexException(){
}

IndexException::IndexException(
	const std::string& function,
	const std::string& where,
	const std::string& message,
	const std::string& hint
) : Exception(function, where, message, hint){
}

IndexException::~IndexException(){
}

};	//End of namespace Ygg
