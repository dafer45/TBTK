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

#include <complex>
#include <sstream>
#include <vector>

namespace TBTK{

class Serializeable{
public:
	/** Serialize object. */
	virtual std::string serialize() const = 0;
protected:
	/** Serialize int. */
	static std::string serialize(int i);

	/** Serialize unsigned int. */
	static std::string serialize(unsigned int u);

	/** Serialize double. */
	static std::string serialize(double d);

	/** Serialize complex<double>. */
	static std::string serialize(std::complex<double> c);
};

inline std::string Serializeable::serialize(int i){
	std::stringstream ss;
	ss << i << " ";

	return ss.str();
}

inline std::string Serializeable::serialize(unsigned int u){
	std::stringstream ss;
	ss << u << " ";

	return ss.str();
}

inline std::string Serializeable::serialize(double d){
	std::stringstream ss;
	ss << d << " ";

	return ss.str();
}

inline std::string Serializeable::serialize(std::complex<double> c){
	std::stringstream ss;
	ss << c << " ";

	return ss.str();
}

};	//End namespace TBTK

#endif
