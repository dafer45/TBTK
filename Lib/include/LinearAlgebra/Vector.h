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
 *  @file Vector.h
 *  @brief General template vector class from which other vectors are derived.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_VECTOR
#define COM_DAFER45_TBTK_VECTOR

#include "TBTKMacros.h"

#include <complex>
#include <initializer_list>

namespace TBTK{

/** Conjugation function for int. */
inline const int& conjugate(const int &i){
	return i;
}

/** Conjugation function for float. */
inline const float& conjugate(const float &f){
	return f;
}

/** Conjugation function for double. */
inline const double& conjugate(const double &d){
	return d;
}

/** Conjugation function for complex<T>. */
template<typename T>
inline const double& conjugate(const std::complex<T> &c){
	return conj(c);
}

template<typename T>
class Vector;

/** Column vector. For compile time optimization. */
template<typename T>
class CVector;

/** Conjugated column vector. For compile time optimization. */
template<typename T>
class CCVector;

/** Negated column vector. For compile time optimization. */
template<typename T>
class NCVector;

/** Conjugated and negated column vector. For compile time optimization. */
template<typename T>
class CNCVector;

/** Row vector. For compile time optimization. */
template<typename T>
class RVector;

/** Conjugated row vector. For compile time optimization. */
template<typename T>
class CRVector;

/** Negated row vector. For compile time optimization. */
template<typename T>
class NRVector;

/** Conjugated and negated row vector. For compile time optimization. */
template<typename T>
class CNRVector;

template<typename T>
class Vector{
public:
	/** Constructor. */
	Vector(unsigned int size);

	/** Constructor. */
	Vector(std::initializer_list<T> components);

	/** Constructor. */
	Vector(const std::vector<T> &components);

	/** Destructor. */
	~Vector();

	/** Get size. */
	unsigned int getSize() const;

	/** Get element at position n. */
	const T& at(unsigned int n) const;

	/** Addition operator. */
	const Vector<T> operator+(const Vector<T> &rhs) const;

	/** Addition operator (CVector = CVector + CVector). */
	friend const CVector<T> operator+(
		const CVector<T> &lhs,
		const CVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] + rhs.components[n];

		return result;
	}

	/** Addition operator (CVector = CVector + CCVector). */
	friend const CVector<T> operator+(
		const CVector<T> &lhs,
		const CCVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] + conjugate(rhs.components[n]);

		return result;
	}

	/** Addition operator (CVector = CVector + NCVector). */
	friend const CVector<T> operator+(
		const CVector<T> &lhs,
		const NCVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] - rhs.components[n];

		return result;
	}

	/** Addition operator (CVector = CVector + CNCVector). */
	friend const CVector<T> operator+(
		const CVector<T> &lhs,
		const CNCVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] - conjugate(rhs.components[n]);

		return result;
	}

	/** Addition operator (CVector = CCVector + CVector). */
	friend const CVector<T> operator+(
		const CCVector<T> &lhs,
		const CVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = conjugate(lhs.components[n]) + rhs.components[n];

		return result;
	}

	/** Addition operator (CCVector = CCVector + CCVector). */
	friend const CCVector<T> operator+(
		const CCVector<T> &lhs,
		const CCVector<T> &rhs
	){
		CCVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] + rhs.components[n];

		return result;
	}

	/** Addition operator (CVector = CCVector + NCVector). */
	friend const CVector<T> operator+(
		const CCVector<T> &lhs,
		const NCVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = conjugate(lhs.components[n]) - rhs.components[n];

		return result;
	}

	/** Addition operator (CCVector = CCVector + CNCVector). */
	friend const CCVector<T> operator+(
		const CCVector<T> &lhs,
		const CNCVector<T> &rhs
	){
		CCVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] - rhs.components[n];

		return result;
	}

	/** Addition operator (CVector = NCVector + CVector). */
	friend const CVector<T> operator+(
		const NCVector<T> &lhs,
		const CVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = -lhs.components[n] + rhs.components[n];

		return result;
	}

	/** Addition operator (CVector = NCVector + CCVector). */
	friend const CVector<T> operator+(
		const NCVector<T> &lhs,
		const CCVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = -lhs.components[n] + conjugate(rhs.components[n]);

		return result;
	}

	/** Addition operator (NCVector = NCVector + NCVector). */
	friend const NCVector<T> operator+(
		const NCVector<T> &lhs,
		const NCVector<T> &rhs
	){
		NCVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] + rhs.components[n];

		return result;
	}

	/** Addition operator (NCVector = NCVector + CNCVector). */
	friend const NCVector<T> operator+(
		const NCVector<T> &lhs,
		const CNCVector<T> &rhs
	){
		NCVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] + conjugate(rhs.components[n]);

		return result;
	}

	/** Addition operator (CVector = CNCVector + CVector). */
	friend const CVector<T> operator+(
		const CNCVector<T> &lhs,
		const CVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = -conjugate(lhs.components[n]) + rhs.components[n];

		return result;
	}

	/** Addition operator (CCVector = CNCVector + CCVector). */
	friend const CCVector<T> operator+(
		const CNCVector<T> &lhs,
		const CCVector<T> &rhs
	){
		CCVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = -lhs.components[n] + rhs.components[n];

		return result;
	}

	/** Addition operator (NCVector = CNCVector + NCVector). */
	friend const NCVector<T> operator+(
		const CNCVector<T> &lhs,
		const NCVector<T> &rhs
	){
		NCVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = conjugate(lhs.components[n]) + rhs.components[n];

		return result;
	}

	/** Addition operator (CNCVector = CNCVector + CNCVector). */
	friend const CNCVector<T> operator+(
		const CNCVector<T> &lhs,
		const CNCVector<T> &rhs
	){
		CNCVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] + rhs.components[n];

		return result;
	}

	/** Subtraction operator. */
	const Vector<T> operator-(const Vector<T> &rhs) const;

	/** Subtraction operator (CVector = CVector - CVector). */
	friend const CVector<T> operator-(
		const CVector<T> &lhs,
		const CVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] - rhs.components[n];

		return result;
	}

	/** Subtraction operator (CVector = CVector - CCVector). */
	friend const CVector<T> operator-(
		const CVector<T> &lhs,
		const CCVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] - conjugate(rhs.components[n]);

		return result;
	}

	/** Addition operator (CVector = CVector - NCVector). */
	friend const CVector<T> operator-(
		const CVector<T> &lhs,
		const NCVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] + rhs.components[n];

		return result;
	}

	/** Addition operator (CVector = CVector - CNCVector). */
	friend const CVector<T> operator-(
		const CVector<T> &lhs,
		const CNCVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] + conjugate(rhs.components[n]);

		return result;
	}

	/** Addition operator (CVector = CCVector - CVector). */
	friend const CVector<T> operator-(
		const CCVector<T> &lhs,
		const CVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = conjugate(lhs.components[n]) - rhs.components[n];

		return result;
	}

	/** Addition operator (CCVector = CCVector - CCVector). */
	friend const CCVector<T> operator-(
		const CCVector<T> &lhs,
		const CCVector<T> &rhs
	){
		CCVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] - rhs.components[n];

		return result;
	}

	/** Addition operator (CVector = CCVector - NCVector). */
	friend const CVector<T> operator-(
		const CCVector<T> &lhs,
		const NCVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = conjugate(lhs.components[n]) + rhs.components[n];

		return result;
	}

	/** Addition operator (CCVector = CCVector - CNCVector). */
	friend const CCVector<T> operator-(
		const CCVector<T> &lhs,
		const CNCVector<T> &rhs
	){
		CCVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] + rhs.components[n];

		return result;
	}

	/** Addition operator (CVector = NCVector - CVector). */
	friend const CVector<T> operator-(
		const NCVector<T> &lhs,
		const CVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = -lhs.components[n] - rhs.components[n];

		return result;
	}

	/** Addition operator (CVector = NCVector - CCVector). */
	friend const CVector<T> operator-(
		const NCVector<T> &lhs,
		const CCVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = -lhs.components[n] - conjugate(rhs.components[n]);

		return result;
	}

	/** Addition operator (NCVector = NCVector - NCVector). */
	friend const NCVector<T> operator-(
		const NCVector<T> &lhs,
		const NCVector<T> &rhs
	){
		NCVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] - rhs.components[n];

		return result;
	}

	/** Addition operator (NCVector = NCVector - CNCVector). */
	friend const NCVector<T> operator-(
		const NCVector<T> &lhs,
		const CNCVector<T> &rhs
	){
		NCVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] - conjugate(rhs.components[n]);

		return result;
	}

	/** Addition operator (CVector = CNCVector - CVector). */
	friend const CVector<T> operator-(
		const CNCVector<T> &lhs,
		const CVector<T> &rhs
	){
		CVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = -conjugate(lhs.components[n]) - rhs.components[n];

		return result;
	}

	/** Addition operator (CCVector = CNCVector - CCVector). */
	friend const CCVector<T> operator-(
		const CNCVector<T> &lhs,
		const CCVector<T> &rhs
	){
		CCVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = -lhs.components[n] - rhs.components[n];

		return result;
	}

	/** Addition operator (NCVector = CNCVector - NCVector). */
	friend const NCVector<T> operator-(
		const CNCVector<T> &lhs,
		const NCVector<T> &rhs
	){
		NCVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = conjugate(lhs.components[n]) - rhs.components[n];

		return result;
	}

	/** Addition operator (CNCVector = CNCVector - CNCVector). */
	friend const CNCVector<T> operator-(
		const CNCVector<T> &lhs,
		const CNCVector<T> &rhs
	){
		CNCVector<T> result(lhs.size);

		for(unsigned int n = 0; n < lhs.size; n++)
			result.components[n] = lhs.components[n] - rhs.components[n];

		return result;
	}

	/** Inversion operator. */
	const Vector<T> operator-() const;

	/** Multiplication operator (Vector*scalar). */
	const Vector<T> operator*(const T &rhs) const;

	/** Multiplication operator (scalar*Vector). */
	friend const Vector<T> operator*(const T &lhs, const Vector<T> &rhs){
		Vector<T> result(rhs.size);

		for(unsigned int n = 0; n < rhs.size; n++)
			result.components[n] = lhs*rhs.components[n];

		return result;
	}

	/** Multiplication operator (scalar*CVector). */
	friend const CVector<T> operator*(const T &lhs, const CVector<T> &rhs){
		CVector<T> result(rhs.size);

		for(unsigned int n = 0; n < rhs.size; n++)
			result.components[n] = lhs*rhs.components[n];

		return result;
	}

	/** Multiplication operator (scalar*CCVector). */
	friend const CCVector<T> operator*(const T &lhs, const CCVector<T> &rhs){
		CCVector<T> result(rhs.size);

		for(unsigned int n = 0; n < rhs.size; n++)
			result.components[n] = conj(lhs)*rhs.components[n];

		return result;
	}

	/** Multiplication operator (scalar*NCVector). */
	friend const NCVector<T> operator*(const T &lhs, const NCVector<T> &rhs){
		NCVector<T> result(rhs.size);

		for(unsigned int n = 0; n < rhs.size; n++)
			result.components[n] = lhs*rhs.components[n];

		return result;
	}

	/** Multiplication operator (scalar*CNCVector). */
	friend const CNCVector<T> operator*(const T &lhs, const CNCVector<T> &rhs){
		CNCVector<T> result(rhs.size);

		for(unsigned int n = 0; n < rhs.size; n++)
			result.components[n] = conj(-lhs)*rhs.components[n];

		return result;
	}
protected:
	/** Components. */
	T* components;
private:
	/** Number of elements. */
	unsigned int size;
};

template<typename T>
Vector<T>::Vector(unsigned int size){
	this->size = size;
	components = new T[size];
}

template<typename T>
Vector<T>::Vector(std::initializer_list<T> components){
	size = components.size();
	this->components = new T[size];
	for(unsigned int n = 0; n < components.size(); n++)
		this->components[n] = *(components.begin()+n);
}

template<typename T>
Vector<T>::Vector(const std::vector<T> &components){
	size = components.size();
	this->components = new T[size];
	for(unsigned int n = 0; n < components.size(); n++)
		this->components[n] = components.at(n);
}

template<typename T>
Vector<T>::~Vector(){
	delete [] components;
}

template<typename T>
unsigned int Vector<T>::getSize() const{
	return size;
}

template<typename T>
const T& Vector<T>::at(unsigned int n) const{
	return components[n];
}

template<typename T>
const Vector<T> Vector<T>::operator+(const Vector<T> &rhs) const{
	Vector<T> result(size);

	TBTKAssert(
		size == rhs.size,
		"Vector<T>::operator+()",
		"Cannot add vectors of different size.",
		""
	);

	for(unsigned int n = 0; n < size; n++)
		result.components[n] = components[n] + rhs.components[n];

	return result;
}

template<typename T>
const Vector<T> Vector<T>::operator-(const Vector<T> &rhs) const{
	Vector<T> result(size);

	TBTKAssert(
		size == rhs.size,
		"Vector<T>::operator-()",
		"Cannot subtract vectors of different size.",
		""
	);

	for(unsigned int n = 0; n < size; n++)
		result.components[n] = components[n] - rhs.components[n];

	return result;
}

template<typename T>
const Vector<T> Vector<T>::operator-() const{
	Vector<T> result(size);

	for(unsigned int n = 0; n < size; n++)
		result.components[n] = -components[n];

	return result;
}

template<typename T>
const Vector<T> Vector<T>::operator*(const T &rhs) const{
	Vector<T> result(size);

	for(unsigned int n = 0; n < size; n++)
		result.components[n] = components[n]*rhs;

	return result;
}

};	//End namespace TBTK

#endif
