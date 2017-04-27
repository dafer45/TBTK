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
 *  @file VectorNd.h
 *  @brief N-dimensional vector with components of double type.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_VECTOR_ND
#define COM_DAFER45_TBTK_VECTOR_ND

#include <cmath>
#include <initializer_list>
#include <ostream>
#include <vector>

namespace TBTK{

class VectorNd{
public:
	/** Constructor. */
	VectorNd(unsigned int size);

	/** Copy constructor. */
	VectorNd(const VectorNd &vectorNd);

	/** Move constructor. */
	VectorNd(VectorNd &&vectorNd);

	/** Constructor. */
	VectorNd(std::initializer_list<double> components);

	/** Constructor. */
	VectorNd(const std::vector<double> &components);

	/** Destructor. */
	~VectorNd();

	/** Assignment operator. */
	VectorNd& operator=(const VectorNd &vectorNd);

	/** Move assignment operator. */
	VectorNd& operator=(VectorNd &&vectorNd);

	/** Addition operator. */
	const VectorNd operator+(const VectorNd &rhs) const;

	/** Subtraction operator. */
	const VectorNd operator-(const VectorNd &rhs) const;

	/** Inversion operator. */
	const VectorNd operator-() const;

	/** Multiplication operator (vector*scalar). */
	const VectorNd operator*(double rhs) const;

	/** Multiplication operator (scalar*vector). */
	friend const VectorNd operator*(double lhs, const VectorNd &rhs);

	/** Division operator. */
	const VectorNd operator/(double rhs) const;

	/** Returns a unit vector pointing in the same direction as the
	 *  original vector. */
	const VectorNd unit() const;

	/** Returns a vector that is the component of the vector that is
	 *  parallel to the argument. */
	const VectorNd parallel(const VectorNd &v) const;

	/** Norm. */
	double norm() const;

	/** Dot product. */
	static double dotProduct(const VectorNd &lhs, const VectorNd &rhs);

	/** Get a std::vector<double> representation of the vector. */
	const std::vector<double> getStdVector() const;

	/** operator<< for ostream. */
	friend std::ostream& operator<<(std::ostream &stream, const VectorNd &v);
private:
	/** Data size. */
	unsigned int size;

	/** Data. */
	double *data;
};

inline const VectorNd VectorNd::operator+(const VectorNd &rhs) const{
	TBTKAssert(
		size == rhs.size,
		"VectorNd::operator+()",
		"Incompatible dimensions. Left hand side has " << size
		<< " components, while the right hand side has " << rhs.size
		<< " components.",
		""
	);

	VectorNd result(size);
	for(unsigned int n = 0; n < size; n++)
		result.data[n] = data[n] + rhs.data[n];

	return result;
}

inline const VectorNd VectorNd::operator-(const VectorNd &rhs) const{
	TBTKAssert(
		size == rhs.size,
		"VectorNd::operator-()",
		"Incompatible dimensions. Left hand side has " << size
		<< " components, while the right hand side has " << rhs.size
		<< " components.",
		""
	);

	VectorNd result(size);
	for(unsigned int n = 0; n < size; n++)
		result.data[n] = data[n] - rhs.data[n];

	return result;
}

inline const VectorNd VectorNd::operator-() const{
	VectorNd result(size);
	for(unsigned int n = 0; n < size; n++)
		result.data[n] = -data[n];

	return result;
}

inline const VectorNd VectorNd::operator*(double rhs) const{
	VectorNd result(size);
	for(unsigned int n = 0; n < size; n++)
		result.data[n] = data[n]*rhs;

	return result;
}

inline const VectorNd operator*(double lhs, const VectorNd &rhs){
	VectorNd result(rhs.size);
	for(unsigned int n = 0; n < rhs.size; n++)
		result.data[n] = lhs*rhs.data[n];

	return result;
}

inline const VectorNd VectorNd::operator/(double rhs) const{
	VectorNd result(size);
	for(unsigned int n = 0; n < size; n++)
		result.data[n] = data[n]/rhs;

	return result;
}

inline const VectorNd VectorNd::unit() const{
	return (*this)/norm();
}

inline const VectorNd VectorNd::parallel(const VectorNd &v) const{
	TBTKAssert(
		size == v.size,
		"VectorNd::parallel()",
		"Incompatible dimensions.",
		""
	);

	return dotProduct(*this, v.unit())*v.unit();
}

inline double VectorNd::norm() const{
	return sqrt(dotProduct(*this, *this));
}

inline double VectorNd::dotProduct(const VectorNd &lhs, const VectorNd &rhs){
	TBTKAssert(
		lhs.size == rhs.size,
		"VectorNd::dotProduct()",
		"Incompatible dimensions. Left hand side has " << lhs.size
		<< " components, while the right hand side has " << rhs.size
		<< " components.",
		""
	);

	double dp = 0;
	for(unsigned int n = 0; n < lhs.size; n++)
		dp += lhs.data[n]*rhs.data[n];

	return dp;
}

inline const std::vector<double> VectorNd::getStdVector() const{
	std::vector<double> result;
	for(unsigned int n = 0; n < size; n++)
		result.push_back(data[n]);

	return result;
}

inline std::ostream& operator<<(std::ostream &stream, const VectorNd &v){
	stream << "(";
	for(unsigned int n = 0; n < v.size; n++){
		if(n != 0)
			stream << ", ";
		stream << v.data[n];
	}

	return stream;
}

};	//End namespace TBTK

#endif
