/* Copyright 2019 Kristofer Björnson
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
 *  @file Streamable.h
 *  @brief Abstract base class for classes that can be written to a stream.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_STREAMABLE
#define COM_DAFER45_TBTK_STREAMABLE

#include <string>

namespace TBTK{

class Streamable{
public:
	/** This function should be implemented by the overriding class and
	 *  return a string with characteristic information about the class.
	 *
	 *  @return A string with characteristic information about the class.
	 */
	virtual std::string toString() const = 0;

	/** Writes the Streamables toString()-representation to a stream.
	 *
	 *  @param stream The stream to write to.
	 *  @param streamable The streamable to write. */
	friend std::ostream& operator<<(
		std::ostream &stream,
		const Streamable &streamable
	);
};

inline std::ostream& operator<<(
	std::ostream &stream,
	const Streamable &streamable
){
	stream << streamable.toString();

	return stream;
}

}; //End of namespace TBTK

#endif
