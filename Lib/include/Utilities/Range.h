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
 *  @file Range.h
 *  @brief One-dimensional range.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_RANGE
#define COM_DAFER45_TBTK_RANGE

#include "Serializeable.h"
#include "TBTKMacros.h"

#include "json.hpp"

namespace TBTK{

class Range : public Serializeable{
public:
	/** Constructor. */
	Range(
		double lowerBound,
		double upperBound,
		unsigned int resolution,
		bool includeLowerBound = true,
		bool includeUpperBound = true
	);

	/** Constructor. Constructs the Range from a serialization string. */
	Range(const std::string &serialization, Mode mode);

	/** Destructor. */
	~Range();

	/** Get resolution. */
	unsigned int getResolution() const;

	/** Array subscript operator. */
	double operator[](unsigned int n) const;

	/** Serilaize. */
	virtual std::string serialize(Mode mode) const;
private:
	/** Start point. */
	double start;

	/** Incremental distance. */
	double dx;

	/** Resolution. */
	unsigned int resolution;
};

inline double Range::operator[](unsigned int n) const{
	return start + n*dx;
}

inline unsigned int Range::getResolution() const{
	return resolution;
}

}; //End of namesapce TBTK

#endif
