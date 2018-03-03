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

#ifndef COM_DAFER45_TBTK_VECTOR_3D
#define COM_DAFER45_TBTK_VECTOR_3D

#include <cmath>
#include <initializer_list>
#include <ostream>
#include <vector>

namespace TBTK{

class Vector3d{
public:
	/** x-component. */
	double x;

	/** y-component. */
	double y;

	/** z-component. */
	double z;

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

	/** Inversion operator. */
	const Vector3d operator-() const;

	/** Multiplication operator (cross product). */
	const Vector3d operator*(const Vector3d &rhs) const;

	/** Multiplication operator (vector*scalar). */
	const Vector3d operator*(double rhs) const;

	/** Multiplication operator (scalar*vector). */
	friend const Vector3d operator*(double lhs, const Vector3d &rhs);

	/** Division operator. */
	const Vector3d operator/(double rhs) const;

	/** Returns a unit vector pointing in the same direction as the
	 *  original vector. */
	const Vector3d unit() const;

	/** Returns a vector tha is the component of the vector that is
	 *  perpendicular to the argument. */
	const Vector3d perpendicular(const Vector3d &v) const;

	/** Returns a vector that is the component of the vector that is
	 *  parallel to the argument. */
	const Vector3d parallel(const Vector3d &v) const;

	/** Norm. */
	double norm() const;

	/** Dot product. */
	static double dotProduct(const Vector3d &lhs, const Vector3d &rhs);

	/** Get a std::vector<double> representation of the vector. */
	const std::vector<double> getStdVector() const;

	/** operator<< for ostream. */
	friend std::ostream& operator<<(std::ostream &stream, const Vector3d &v);
};

inline const Vector3d Vector3d::operator+(const Vector3d &rhs) const{
	Vector3d result;

	result.x = x + rhs.x;
	result.y = y + rhs.y;
	result.z = z + rhs.z;

	return result;
}

inline const Vector3d Vector3d::operator-(const Vector3d &rhs) const{
	Vector3d result;

	result.x = x - rhs.x;
	result.y = y - rhs.y;
	result.z = z - rhs.z;

	return result;
}

inline const Vector3d Vector3d::operator-() const{
	Vector3d result;

	result.x = -x;
	result.y = -y;
	result.z = -z;

	return result;
}

inline const Vector3d Vector3d::operator*(const Vector3d &rhs) const{
	Vector3d result;

	result.x = y*rhs.z - z*rhs.y;
	result.y = z*rhs.x - x*rhs.z;
	result.z = x*rhs.y - y*rhs.x;

	return result;
}

inline const Vector3d Vector3d::operator*(double rhs) const{
	Vector3d result;

	result.x = x*rhs;
	result.y = y*rhs;
	result.z = z*rhs;

	return result;
}

inline const Vector3d operator*(double lhs, const Vector3d &rhs){
	Vector3d result;

	result.x = lhs*rhs.x;
	result.y = lhs*rhs.y;
	result.z = lhs*rhs.z;

	return result;
}

inline const Vector3d Vector3d::operator/(double rhs) const{
	Vector3d result;

	result.x = x/rhs;
	result.y = y/rhs;
	result.z = z/rhs;

	return result;
}

inline const Vector3d Vector3d::unit() const{
	return (*this)/norm();
}

inline const Vector3d Vector3d::perpendicular(const Vector3d &v) const{
	return *this - dotProduct(*this, v.unit())*v.unit();
}

inline const Vector3d Vector3d::parallel(const Vector3d &v) const{
	return dotProduct(*this, v.unit())*v.unit();
}

inline double Vector3d::norm() const{
	return sqrt(x*x + y*y + z*z);
}

inline double Vector3d::dotProduct(const Vector3d &lhs, const Vector3d &rhs){
	return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z;
}

inline const std::vector<double> Vector3d::getStdVector() const{
	std::vector<double> result;

	result.push_back(x);
	result.push_back(y);
	result.push_back(z);

	return result;
}

inline std::ostream& operator<<(std::ostream &stream, const Vector3d &v){
	stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";

	return stream;
}

};	//End namespace TBTK

#endif
