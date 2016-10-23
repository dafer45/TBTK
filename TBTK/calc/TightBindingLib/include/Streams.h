/* Copyright 2016 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
	static void setStdMuteOut();

	/** Mute error stream. */
	static void setStdMuteErr();

	/** Open log. */
	static void openLog(std::string filename = "TBTKLog");

	/** Close log. */
	static void closeLog();

	/** Returns true if the standard log file is open. */
	static bool logIsOpen();
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

		/** Mute stream n. */
		void mute(int n, bool isMute);
	private:
		/** Output streams. */
		std::basic_ostream<char, std::char_traits<char>> *ostreams[2];

		/** Flag indicating whether corresponding ostream is muted. */
		bool isMute[2];

		/** Implements std::streambuf::overflow().
		 *  Writes char to ostream1 and ostream2. */
		int overflow(int c);
	};

	/** LogBuffer. */
	class LogBuffer : public std::streambuf{
	public:
		/** Constructor. */
		LogBuffer();

		/** Destructor. */
		~LogBuffer();

		/** Set file output stream. */
		void open(std::string fileName);

		/** Close file output stream. */
		void close();

		/** Returns true if log file is open. */
		bool isOpen();
	private:
		/** File output stream. */
		std::ofstream fout;

		/** Implements std::streambuf::overflow().
		 *  Writes char to ostream1 and ostream2. */
		int overflow(int c);
	};

	/** Null stream for muting. */
	static std::ostream null;

	/** Standard output buffer. */
	static ForkBuffer stdOutBuffer;

	/** Standard log buffer. */
	static LogBuffer stdLogBuffer;

	/** Standard error buffer. */
	static ForkBuffer stdErrBuffer;

	/** Log file. */
	static std::ofstream logFile;
};

};	//End of namespace Util
};	//End of namespace TBTK

#endif

