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

/** Streams for TBTK output.
 *
 *  The Streams class provides global functions for streaming text output. By
 *  default the information is written to stdout and atderr, but the Streams
 *  class allows for more customized output. All text output from TBTK is
 *  written through the Stream interface, and can therefore be customized
 *  through it.
 *
 *  # Streams::out
 *  This is the standard output in TBTK. By default, information written to
 *  Streams::out is forwarded to std::cout and Streams::log.
 *
 *  # Streams::err
 *  This is the error output in TBTK. By default, information written to
 *  Streams::err is forwarded to std::err and Streams::log.
 *
 *  # Streams::log
 *  This is the logged output in TBTK. By default, the log does not write to
 *  anything. However, The two commands
 *  ```cpp
 *    Streams::openLog("LogFilename");
 *  ```
 *  and
 *  ```cpp
 *    Streams::closeLog();
 *  ```
 *  can be used to open and close a log file to which everything is written.
 *
 *  When the log is opened, a time stamp, version number, and git hash for the
 *  currently installed version of TBTK is added to the output. Similarly, a
 *  time stamp is added when the log is closed. This makes it easy to document
 *  the exact setup used to perform a calculation and provides a way for
 *  ensuring that the results will be possible to reproduce any time in the
 *  future. */
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

	/** Open log.
	 *
	 *  @param filename The filename of the log file. */
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

};	//End of namespace TBTK

#endif

