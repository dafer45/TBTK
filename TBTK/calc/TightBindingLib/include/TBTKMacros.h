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

#define TBTKAssert(expression, function, message, hint)	\
	if(!(expression)){	\
		TBTK::Util::Streams::err << "Error in " << function << "\n";	\
		TBTK::Util::Streams::err << "\t" << message << "\n";	\
		if(std::strcmp(hint, "") != 0)	\
			TBTK::Util::Streams::err << "\tHint: " << hint << "\n";	\
		TBTK::Util::Streams::err << "\tWhere: " << __FILE__ << ", " << __LINE__ << "\n";	\
		Util::Streams::closeLog();	\
		exit(1);	\
	}

#define TBTKExit(function, message, hint)	\
	TBTK::Util::Streams::err << "Error in " << function << "\n";	\
	TBTK::Util::Streams::err << "\t" << message << "\n";	\
	if(std::strcmp(hint, "") != 0)	\
		TBTK::Util::Streams::err << "\tHint: " << hint << "\n";	\
	TBTK::Util::Streams::err << "\tWhere: " << __FILE__ << ", " << __LINE__ << "\n";	\
	Util::Streams::closeLog();	\
	exit(1);

#endif
