/* Copyright 2018 Kristofer Björnson
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
 *  @file Vector2d.h
 *  @brief Two-dimensional vector with components of double type.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_VECTOR_2D
#define COM_DAFER45_TBTK_VECTOR_2D

#include <cmath>
#include <initializer_list>
#include <ostream>
#include <vector>

namespace TBTK{

class Vector2d{
public:
	/** x-component. */
	double x;

	/** y-component. */
	double y;

	/** Constructor. */
	Vector2d();

	/** Constructor. */
	Vector2d(std::initializer_list<double> components);

	/** Constructor. */
	Vector2d(const std::vector<double> &components);

	/** Addition operator. */
	const Vector2d operator+(const Vector2d &rhs) const;

	/** Subtraction operator. */
	const Vector2d operator-(const Vector2d &rhs) const;

	/** Inversion operator. */
	const Vector2d operator-() const;

	/** Multiplication operator (vector*scalar). */
	const Vector2d operator*(double rhs) const;

	/** Multiplication operator (scalar*vector). */
	friend const Vector2d operator*(double lhs, const Vector2d &rhs);

	/** Division operator. */
	const Vector2d operator/(double rhs) const;

	/** Returns a unit vector pointing in the same direction as the
	 *  original vector. */
	const Vector2d unit() const;

	/** Returns a vector that is the component of the vector that is
	 *  perpendicular to the argument. */
	const Vector2d perpendicular(const Vector2d &v) const;

	/** Returns a vector that is the component of the vector that is
	 *  parallel to the argument. */
	const Vector2d parallel(const Vector2d &v) const;

	/** Norm. */
	double norm() const;

	/** Dot product. */
	static double dotProduct(const Vector2d &lhs, const Vector2d &rhs);

	/** Get a std::vector<double> representation of the vector. */
	const std::vector<double> getStdVector() const;

	/** operator<< for ostream. */
	friend std::ostream& operator<<(std::ostream &stream, const Vector2d &v);
};

inline const Vector2d Vector2d::operator+(const Vector2d &rhs) const{
	Vector2d result;

	result.x = x + rhs.x;
	result.y = y + rhs.y;

	return result;
}

inline const Vector2d Vector2d::operator-(const Vector2d &rhs) const{
	Vector2d result;

	result.x = x - rhs.x;
	result.y = y - rhs.y;

	return result;
}

inline const Vector2d Vector2d::operator-() const{
	Vector2d result;

	result.x = -x;
	result.y = -y;

	return result;
}

inline const Vector2d Vector2d::operator*(double rhs) const{
	Vector2d result;

	result.x = x*rhs;
	result.y = y*rhs;

	return result;
}

inline const Vector2d operator*(double lhs, const Vector2d &rhs){
	Vector2d result;

	result.x = lhs*rhs.x;
	result.y = lhs*rhs.y;

	return result;
}

inline const Vector2d Vector2d::operator/(double rhs) const{
	Vector2d result;

	result.x = x/rhs;
	result.y = y/rhs;

	return result;
}

inline const Vector2d Vector2d::unit() const{
	return (*this)/norm();
}

inline const Vector2d Vector2d::perpendicular(const Vector2d &v) const{
	return *this - dotProduct(*this, v.unit())*v.unit();
}

inline const Vector2d Vector2d::parallel(const Vector2d &v) const{
	return dotProduct(*this, v.unit())*v.unit();
}

inline double Vector2d::norm() const{
	return sqrt(x*x + y*y);
}

inline double Vector2d::dotProduct(const Vector2d &lhs, const Vector2d &rhs){
	return lhs.x*rhs.x + lhs.y*rhs.y;
}

inline const std::vector<double> Vector2d::getStdVector() const{
	std::vector<double> result;

	result.push_back(x);
	result.push_back(y);

	return result;
}

inline std::ostream& operator<<(std::ostream &stream, const Vector2d &v){
	stream << "(" << v.x << ", " << v.y << ")";

	return stream;
}

};	//End namespace TBTK

#endif
