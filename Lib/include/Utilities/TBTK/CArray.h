/* Copyright 2019 Kristofer Björnson
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
 *  @file CArray.h
 *  @brief Container for a C style array.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_C_ARRAY
#define COM_DAFER45_TBTK_C_ARRAY

#include "TBTK/TBTKMacros.h"

#include "TBTK/json.hpp"

namespace TBTK{

template<typename DataType>
class CArray{
public:
	//TBTKFeature Utilities.CArray.construction.1 2019-10-30
	/** Constructor. */
	CArray();

	//TBTKFeature Utilities.CArray.construction.2 2019-10-30
	/** Constructor.
	 *
	 *  @param size The size of the array. */
	CArray(unsigned int size);

	//TBTKFeature Utilities.CArray.copyConstruction.1.C++ 2019-10-30
	/** Copy constructor.
	 *
	 *  @param carray The carray to copy. */
	CArray(const CArray &carray);

	//TBTKFeature Utilities.CArray.moveConstruction.1.C++ 2019-10-30
	/** Move constructor.
	 *
	 *  @param carray The carray to move. */
	CArray(CArray &&carray);

	/** Destructor. */
	~CArray();

	/** Assignment operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The left hand side after assignment has occured. */
	CArray& operator=(const CArray &carray);

	/** Move assignment operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The left hand side after assignment has occured. */
	CArray& operator=(CArray &&carray);

	//TBTKFeature Utilities.CArray.operatorArraySubscript.1 2019-10-30
	/** Array subscript operator.
	 *
	 *  @param n Position to get the value for.
	 *
	 *  @return The value at position n. */
	DataType& operator[](unsigned int n);

	//TBTKFeature Utilities.CArray.operatorArraySubscript.2 2019-10-30
	/** Array subscript operator.
	 *
	 *  @param n Position to get the value for.
	 *
	 *  @return The value at position n. */
	const DataType& operator[](unsigned int n) const;

	/** Get size.
	 *
	 *  @return The size of the array. */
	unsigned int getSize() const;
private:
	/** Size. */
	unsigned int size;

	/** Data. */
	DataType *data;
};

template<typename DataType>
CArray<DataType>::CArray(){
	data = nullptr;
}

template<typename DataType>
CArray<DataType>::CArray(unsigned int size){
	this->size = size;
	data = new DataType[size];
}

template<typename DataType>
CArray<DataType>::CArray(const CArray &carray){
	size = carray.size;
	if(carray.data == nullptr){
		data = nullptr;
	}
	else{
		data = new DataType[size];
		for(unsigned int n = 0; n < size; n++)
			data[n] = carray.data[n];
	}
}

template<typename DataType>
CArray<DataType>::CArray(CArray &&carray){
	size = carray.size;
	if(carray.data == nullptr){
		data = nullptr;
	}
	else{
		data = carray.data;
		carray.data = nullptr;
	}
}

template<typename DataType>
CArray<DataType>::~CArray(){
	if(data != nullptr)
		delete [] data;
}

template<typename DataType>
CArray<DataType>& CArray<DataType>::operator=(const CArray &rhs){
	if(this != &rhs){
		size = rhs.size;
		if(data != nullptr)
			delete [] data;

		if(rhs.data == nullptr){
			data = nullptr;
		}
		else{
			data = new DataType[size];
			for(unsigned int n = 0; n < size; n++)
				data[n] = rhs.data[n];
		}
	}

	return *this;
}

template<typename DataType>
CArray<DataType>& CArray<DataType>::operator=(CArray &&rhs){
	if(this != &rhs){
		size = rhs.size;
		if(data != nullptr)
			delete [] data;

		if(rhs.data == nullptr){
			data = nullptr;
		}
		else{
			data = rhs.data;
			rhs.data = nullptr;
		}
	}

	return *this;
}

template<typename DataType>
DataType& CArray<DataType>::operator[](unsigned int n){
	return data[n];
}

template<typename DataType>
const DataType& CArray<DataType>::operator[](unsigned int n) const{
	return data[n];
}

template<typename DataType>
unsigned int CArray<DataType>::getSize() const{
	return size;
}

}; //End of namesapce TBTK

#endif
