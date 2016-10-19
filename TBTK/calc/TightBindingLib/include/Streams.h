/** @package TBTKcalc
 *  @file Streams.h
 *  @brief Streams for TBTK output.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_STREAMS
#define COM_DAFER45_TBTK_STREAMS

#include <ostream>

namespace TBTK{
namespace Util{

class Streams{
public:
	/** Standard output stream. */
	static std::ostream &out;

	/** Detailed log output stream. */
	static std::ostream &log;

	/** Error output stream. */
	static std::ostream &err;
private:
};

};	//End of namespace Util
};	//End of namespace TBTK

#endif
