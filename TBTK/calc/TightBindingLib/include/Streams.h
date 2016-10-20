/** @package TBTKcalc
 *  @file Streams.h
 *  @brief Streams for TBTK output.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_STREAMS
#define COM_DAFER45_TBTK_STREAMS

#include <ostream>
#include <fstream>
#include <string>
#include <vector>

namespace TBTK{
namespace Util{

class Streams{
public:
	/** Standard output stream. */
	static std::ostream out;

	/** Detailed log output stream. */
	static std::ostream log;

	/** Error output stream. */
	static std::ostream err;

	/** Mute output stream. */
	static void muteOut();

	/** Mute log stream. */
	static void muteLog();

	/** Mute error stream. */
	static void muteErr();
private:
	/** Null buffer for muting. */
	static class NullBuffer : public std::streambuf{
	public:
		int overflow(int c);
	} nullBuffer;

	/** Fork buffer. */
	class ForkBuffer : public std::streambuf{
	public:
		/** Constructor. */
		ForkBuffer(
			std::basic_ostream<char, std::char_traits<char>> *ostream1,
			std::basic_ostream<char, std::char_traits<char>> *ostream2
		);
	private:
		/** First output stream. */
		std::basic_ostream<char, std::char_traits<char>> *ostream1;

		/** Second output stream. */
		std::basic_ostream<char, std::char_traits<char>> *ostream2;

		/** Implements std::streambuf::overflow().
		 *  Writes char to ostream1 and ostream2. */
		int overflow(int c);
	};

	/** Null stream for muting. */
	static std::ostream null;
};

};	//End of namespace Util
};	//End of namespace TBTK

#endif

