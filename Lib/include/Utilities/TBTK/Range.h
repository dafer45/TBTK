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

#include "TBTK/Serializable.h"
#include "TBTK/TBTKMacros.h"

#include "TBTK/json.hpp"

namespace TBTK{

/** @brief One-dimensional range.
 *
 *  The Range can be used to define a one-dimensional interval. It has a lower
 *  and upper bound, an is resolved with a number of points. A Range is created
 *  using
 *  ```cpp
 *    Range range(LOWER_BOUND, UPPER_BOUND, RESOLUTION);
 *  ```
 *
 *  # Bounds
 *  By default the bounds are included in the Range. But it is possible to
 *  exclude one or both bounds by passing two add boolean flags to the
 *  constructor. The first and second flag indicates whether the lower and
 *  upper bounds are included, respectively.
 *
 *  # Example
 *  \snippet Utilities/Range.cpp Range
 *  ## Output
 *  \snippet output/Utilities/Range.output Range */
class Range : public Serializable{
public:
	/** Constructor.
	 *
	 *  @param lowerBound The lower bound of the range.
	 *  @param TupperBound The upper bound of the range.
	 *  @param resolution The number of points with which to resolve the
	 *  range.
	 *
	 *  @param includeLowerBound Flag indicating whether or not the lower
	 *  bound should be included.
	 *
	 *  @param includeUpperBound Flag indicating whether or not the upper
	 *  bound should be included. */
	Range(
		double lowerBound,
		double upperBound,
		unsigned int resolution,
		bool includeLowerBound = true,
		bool includeUpperBound = true
	);

	/** Constructor. Constructs the Range from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Range.
	 *
	 *  @param mode The mode with which the string has been serialized. */
	Range(const std::string &serialization, Mode mode);

	/** Get resolution.
	 *
	 *  @return The number of points with which the range is resolved. */
	unsigned int getResolution() const;

	/** Array subscript operator.
	 *
	 *  @param n The index of the entry to return.
	 *
	 *  @return The nth entry in the Range. */
	double operator[](unsigned int n) const;

	/** Serilaize.
	 *
	 *  @param mode The mode to use.
	 *
	 *  @return Serialized string representation of the Range. */
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
