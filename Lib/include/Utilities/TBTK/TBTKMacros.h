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
 *  @file TBTKMacros.h
 *  @brief Precompiler macros
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_MACRO
#define COM_DAFER45_TBTK_MACRO

#include "TBTK/Streams.h"

#include <chrono>
#include <cstring>
#include <ctime>
#include <sstream>
#include <string>

#define TBTK_VERSION_STRING std::to_string(TBTK_VERSION_MAJOR) + "." \
	+ std::to_string(TBTK_VERSION_MINOR) + "." \
	+ std::to_string(TBTK_VERSION_PATCH)

#define TBTK_ABOUT_STRING \
	"TBTK\n" \
	"Version:\t" + TBTK_VERSION_STRING + "\n" \
	"Git hash:\t" TBTK_VERSION_GIT_HASH

inline std::string TBTK_GET_CURRENT_TIME_STRING(){
	std::chrono::time_point<std::chrono::system_clock> timePoint
		= std::chrono::system_clock::now();
	std::time_t now = std::chrono::system_clock::to_time_t(timePoint);

	return std::ctime(&now);
}

#define TBTK_RUNTIME_CONTEXT_STRING \
	TBTK_ABOUT_STRING + "\n" \
	+ "Date:\t" + TBTK_GET_CURRENT_TIME_STRING()

#ifdef TBTKOptimize
	#define TBTKAssert(expression, function, message, hint)	;
	#define TBTKExceptionAssert(expression, exception);
	#define TBTKExit(function, message, hint) exit(1);
#else
	#define TBTKAssert(expression, function, message, hint)	\
		if(!(expression)){	\
			TBTK::Streams::err << "Error in " << function << "\n";	\
			TBTK::Streams::err << "\t" << message << "\n";	\
			std::stringstream hintStream;	\
			hintStream << hint;	\
			if(std::strcmp(hintStream.str().c_str(), "") != 0)	\
				TBTK::Streams::err << "\tHint: " << hint << "\n";	\
			TBTK::Streams::err << "\tWhere: " << __FILE__ << ", " << __LINE__ << "\n";	\
			if(TBTK::Streams::logIsOpen())	\
				TBTK::Streams::closeLog();	\
			exit(1);	\
		}

	#define TBTKExceptionAssert(expression, exception)	\
		if(!(expression))	\
			throw exception;

	#define TBTKExit(function, message, hint)	\
		TBTK::Streams::err << "Error in " << function << "\n";	\
		TBTK::Streams::err << "\t" << message << "\n";	\
		std::stringstream hintStream;	\
		hintStream << hint;	\
		if(std::strcmp(hintStream.str().c_str(), "") != 0)	\
			TBTK::Streams::err << "\tHint: " << hint << "\n";	\
		TBTK::Streams::err << "\tWhere: " << __FILE__ << ", " << __LINE__ << "\n";	\
		if(TBTK::Streams::logIsOpen())	\
			TBTK::Streams::closeLog();	\
		exit(1);
#endif

#define TBTKNotYetImplemented(function)	\
	TBTK::Streams::err << "Error in " << function << "\n";	\
	TBTK::Streams::err << "\tNot yet implemented.\n";	\
	TBTK::Streams::err << "\tWhere: " << __FILE__ << ", " << __LINE__ << "\n";	\
	if(TBTK::Streams::logIsOpen())	\
		TBTK::Streams::closeLog();	\
	exit(1);

#define TBTKReadableCodeBlock(code) ;

#define TBTKWhere std::string(__FILE__) + ", " + std::to_string(__LINE__)

#endif
