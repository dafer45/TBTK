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
 *  @file ParametrizedLine3d.h
 *  @brief Parametrized line in three dimensions.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PARAMETRIZED_LINE_3D
#define COM_DAFER45_TBTK_PARAMETRIZED_LINE_3D

#include "Field.h"
#include "Vector3d.h"

#include <initializer_list>

namespace TBTK{

/** ParametrizedLine3d is a parametrized line of the form
 *      start + lambda*direction,
 *  where start and direction are three-dimensional vectors (Vector3d), while
 *  lambda is a free parameter. */
class ParametrizedLine3d : public Field<Vector3d, double>{
public:
	/** Constructor. */
	ParametrizedLine3d(const Vector3d &start, const Vector3d &direction);

	/** Implements Field::operator(). */
	virtual Vector3d operator()(std::initializer_list<double> lambda) const;

	/** Mnemoic for operator()(std::initializer_list<double> &lambda). */
	const Vector3d operator()(double lambda) const;
private:
	/** Start point. */
	Vector3d start;

	/** Direction. */
	Vector3d direction;
};

inline const Vector3d ParametrizedLine3d::operator()(double lambda) const{
	return ParametrizedLine3d::operator()({lambda});
}

};	//End namespace TBTK

#endif
