#define TBTK_NUMPY_INITIALIZING_TRANSLATION_UNIT
#include "TBTK/External/MatPlotLibCpp/matplotlibcpp.h"

namespace matplotlibcpp{
namespace detail{

std::string s_backend;

/* For now, _interpreter is implemented as a singleton since its currently not possible to have
  multiple independent embedded python interpreters without patching the python source code
  or starting a separate process for each.
  http://bytes.com/topic/python/answers/793370-multiple-independent-python-interpreters-c-c-program
 */
_interpreter& _interpreter::get(){
	static _interpreter ctx;
	return ctx;
}

#ifndef WITHOUT_NUMPY
#	if PY_MAJOR_VERSION >= 3
void* _interpreter::import_numpy(){
	import_array();
	return NULL;
}
#	else
void _interpreter::import_numpy(){
	import_array();
}
#	endif
#endif

};	//End of namespace detail
};	//End of namespace matplotlibcpp
