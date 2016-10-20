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
	class DynamicOstream{
	public:
		/** Constructor. */
		DynamicOstream(std::ostream *streamPointer);

		/** Destructor. */
		~DynamicOstream();

		/** Operator << for bool. */
		std::ostream& operator<<(bool val);

		/** Operator << for short. */
		std::ostream& operator<<(short val);

		/** Operator << for unsigned short. */
		std::ostream& operator<<(unsigned short val);

		/** Operator << for int. */
		std::ostream& operator<<(int val);

		/** Operator << for unsigned int. */
		std::ostream& operator<<(unsigned int val);

		/** Operator << for long. */
		std::ostream& operator<<(long val);

		/** Operator << for unsigned long. */
		std::ostream& operator<<(unsigned long val);

		/** Operator << for long long. */
		std::ostream& operator<<(long long val);

		/** Operator << for unsigned long long. */
		std::ostream& operator<<(unsigned long long val);

		/** Operator << for float. */
		std::ostream& operator<<(float val);

		/** Operator << for double. */
		std::ostream& operator<<(double val);

		/** Operator << for long double. */
		std::ostream& operator<<(long double val);

		/** Operator << for void*. */
		std::ostream& operator<<(void* val);

		/** Operator << for std::streambuf*. */
		std::ostream& operator<<(std::streambuf* bf);

		/** Operator << for std::ostream& (*pf)(std::ostream&). */
		std::ostream& operator<<(std::ostream& (*pf)(std::ostream&));

		/** Operator << for std::ios& (*pf)(std::ios&). */
		std::ostream& operator<<(std::ios& (*pf)(std::ios&));

		/** Operator << for std::ios_base& (*pf)(std::ios_base&). */
		std::ostream& operator<<(std::ios_base& (*pf)(std::ios_base&));

		/** Operator << for const char*. */
		std::ostream& operator<<(const char *c);

		void setStream(std::ostream *streamPointer);
	private:
		std::ostream *streamPointer;

		friend class Streams;
	};

	/** Standard output stream. */
//	static std::ostream &out;
	static DynamicOstream out;

	/** Detailed log output stream. */
//	static std::ostream &log;
	static DynamicOstream log;

	/** Error output stream. */
//	static std::ostream &err;
	static DynamicOstream err;

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

	/** Null stream for muting. */
	static std::ostream null;
};

inline void Streams::DynamicOstream::setStream(std::ostream *streamPointer){
	this->streamPointer = streamPointer;
}

};	//End of namespace Util
};	//End of namespace TBTK

#endif

