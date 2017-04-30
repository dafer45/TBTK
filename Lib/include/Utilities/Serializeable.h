/* Copyright 2017 Kristofer Björnson
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
 *  @file Serializeable.h
 *  @brief Abstract base class for serializeable objects.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SERIALIZEABLE
#define COM_DAFER45_TBTK_SERIALIZEABLE

#include "TBTKMacros.h"

#include <complex>
#include <sstream>
#include <vector>

namespace TBTK{

class Serializeable{
public:
	/** Serializeation modes. Note that debug is not guaranteed to be
	 *  backward compatible. */
	enum class Mode {Debug, Binary, XML, JSON};

	/** Serialize object. */
	virtual std::string serialize(Mode mode) const = 0;

	/** Validate serialization string. */
	static bool validate(
		const std::string &serialization,
		const std::string &id,
		Mode mode
	);

	/** Get the ID of a serialization string. */
	static std::string getID(const std::string &serialization, Mode mode);

	/** Get the content of a serializtion string. */
	static std::string getContent(
		const std::string &serialization,
		Mode mode
	);

	/** Split content string. */
	static std::vector<std::string> split(
		const std::string &content,
		Mode mode
	);
protected:
	/** Serialize int. */
	static std::string serialize(int i, Mode mode);

	/** Serialize unsigned int. */
	static std::string serialize(unsigned int u, Mode mode);

	/** Serialize double. */
	static std::string serialize(double d, Mode mode);

	/** Serialize complex<double>. */
	static std::string serialize(std::complex<double> c, Mode mode);
};

inline std::string Serializeable::serialize(int i, Mode mode){
	switch(mode){
	case Mode::Debug:
	{
		std::stringstream ss;
		ss << i << " ";

		return ss.str();
	}
	default:
		TBTKExit(
			"Serializeable::serialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

inline std::string Serializeable::serialize(unsigned int u, Mode mode){
	switch(mode){
	case Mode::Debug:
	{
		std::stringstream ss;
		ss << u << " ";

		return ss.str();
	}
	default:
		TBTKExit(
			"Serializeable::serialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

inline std::string Serializeable::serialize(double d, Mode mode){
	switch(mode){
	case Mode::Debug:
	{
		std::stringstream ss;
		ss << d << " ";

		return ss.str();
	}
	default:
		TBTKExit(
			"Serializeable::serialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

inline std::string Serializeable::serialize(std::complex<double> c, Mode mode){
	switch(mode){
	case Mode::Debug:
	{
		std::stringstream ss;
		ss << c << " ";

		return ss.str();
	}
	default:
		TBTKExit(
			"Serializeable::serialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

};	//End namespace TBTK

#endif
