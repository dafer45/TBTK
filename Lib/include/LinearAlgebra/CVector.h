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
 *  @file CVector.h
 *  @brief Template for column vector class.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_CVECTOR
#define COM_DAFER45_TBTK_CVECTOR

#include "Vector.h"

#include <initializer_list>

namespace TBTK{

template<typename T>
class CVector : public Vector<T>{
public:
	/** Constructor. */
	CVector(unsigned int size);

	/** Constructor. */
	CVector(std::initializer_list<T> components);

	/** Constructor. */
	CVector(const std::vector<T> &components);

	/** Destructor. */
	~CVector();

	/** Get element at position n. */
	const T& at(unsigned int n) const;
};

template<typename T>
CVector<T>::CVector(unsigned int size) : Vector<T>(size){
}

template<typename T>
CVector<T>::CVector(std::initializer_list<T> components) : Vector<T>(components){
}

template<typename T>
CVector<T>::CVector(const std::vector<T> &components) : Vector<T>(components){
}

template<typename T>
CVector<T>::~CVector(){
}

template<typename T>
const T& CVector<T>::at(unsigned int n) const{
	return Vector<T>::components[n];
}

};	//End of namespace TBTK

#endif
