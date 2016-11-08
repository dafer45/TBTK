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
 *  @file Vector3d.h
 *  @brief Three-dimensional vector with components of double type.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_VECTOR
#define COM_DAFER45_TBTK_VECTOR

#include <initializer_list>
#include <vector>

namespace TBTK{

/** Container for density of states (DOS). */
class Vector3d{
public:
	/** x-component. */
	int x;

	/** y-component. */
	int y;

	/** z-component. */
	int z;

	/** Constructor. */
	Vector3d();

	/** Constructor. */
	Vector3d(std::initializer_list<double> components);

	/** Constructor. */
	Vector3d(const std::vector<double> &components);

	/** Destructor. */
	~Vector3d();

	/** Addition operator. */
	const Vector3d operator+(const Vector3d &rhs) const;

	/** Subtraction operator. */
	const Vector3d operator-(const Vector3d &rhs) const;

	/** Multiplication operator (cross product). */
	const Vector3d operator*(const Vector3d &rhs) const;
};

inline const Vector3d Vector3d::operator+(const Vector3d &rhs) const{
	Vector3d result;

	result.x = this->x + rhs.x;
	result.y = this->y + rhs.y;
	result.z = this->z + rhs.z;

	return result;
}

inline const Vector3d Vector3d::operator-(const Vector3d &rhs) const{
	Vector3d result;

	result.x = this->x - rhs.x;
	result.y = this->y - rhs.y;
	result.z = this->z - rhs.z;

	return result;
}

inline const Vector3d Vector3d::operator*(const Vector3d &rhs) const{
	Vector3d result;

	result.x = this->y*rhs.z - this->z*rhs.y;
	result.y = this->z*rhs.x - this->x*rhs.z;
	result.z = this->x*rhs.y - this->y*rhs.x;

	return result;
}

};	//End namespace TBTK

#endif
