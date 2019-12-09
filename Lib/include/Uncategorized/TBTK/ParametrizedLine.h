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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file ParametrizedLine.h
 *  @brief Parametrized line.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PARAMETRIZED_LINE
#define COM_DAFER45_TBTK_PARAMETRIZED_LINE

#include "TBTK/Field.h"

#include <initializer_list>
#include <vector>

namespace TBTK{

/** ParametrizedLine is a parametrized line of the form
 *      start + lambda*direction,
 *  where start and direction are n-dimensional vectors, while lambda is a
 *  free parameter. */
class ParametrizedLine : public Field<std::vector<double>, double>{
public:
	/** Constructor. */
	ParametrizedLine(
		std::initializer_list<double> start,
		std::initializer_list<double> direction
	);

	/** Constructor. */
	ParametrizedLine(
		const std::vector<double> &start,
		const std::vector<double> &direction
	);

	/** Implements Field::operator(). */
	virtual std::vector<double> operator()(
		std::initializer_list<double> lambda
	) const;

	/** Mnemoic for operator()(std::initializer_list<double> &lambda). */
	const std::vector<double> operator()(double lambda) const;
private:
	/** Start point. */
	std::vector<double> start;

	/** Direction. */
	std::vector<double> direction;
};

inline const std::vector<double> ParametrizedLine::operator()(
	double lambda
) const{
	return ParametrizedLine::operator()({lambda});
}

};	//End namespace TBTK

#endif
/// @endcond
