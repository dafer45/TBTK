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

#include "Streams.h"

#include <cstring>
#include <sstream>

#ifdef TBTKOptimize
	#define TBTKAssert(expression, function, message, hint)	;
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

#endif
