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
 *  @file Array.h
 *  @brief Multi-dimensional array.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ARRAY
#define COM_DAFER45_TBTK_ARRAY

#include "TBTKMacros.h"

#include <vector>

namespace TBTK{

template<typename DataType>
class Array{
public:
	/** Constructor. */
	Array(const std::vector<unsigned int> &ranges);

	/** Destructor. */
	~Array();

	/** Array subscript operator. */
	DataType& operator[](const std::initializer_list<unsigned int> &index);

	/** Array subscript operator. */
	const DataType& operator[](const std::initializer_list<unsigned int> &index) const;
private:
	/** Data data. */
	DataType *data;

	/** Data size. */
	unsigned int size;

	/** Ranges. */
	std::vector<unsigned int> ranges;
};

template<typename DataType>
Array<DataType>::Array(const std::vector<unsigned int> &ranges){
	this->ranges = ranges;
	size = 1;
	for(unsigned int n = 0; n < ranges.size(); n++){
		TBTKAssert(
			ranges[n] > 0,
			"Array::Array()",
			"Invalid ranges.",
			"'ranges' must only contain positive numbers."
		);
		size *= ranges[n];
	}

	data = new DataType[size];
}

template<typename DataType>
Array<DataType>::~Array(){
	delete [] data;
}

template<typename DataType>
DataType& Array<DataType>::operator[](
	const std::initializer_list<unsigned int> &index
){
	unsigned int idx = 0;
	for(unsigned int n = 0; n < index.size(); n++){
		if(n != 0)
			idx *= ranges[n];
		idx += *(index.begin() + n);
	}

	return data[idx];
}

template<typename DataType>
const DataType& Array<DataType>::operator[](
	const std::initializer_list<unsigned int> &index
) const{
	unsigned int idx = 0;
	for(unsigned int n = 0; n < index.size(); n++){
		if(n != 0)
			idx *= ranges[n];
		idx += *(index.begin() + n);
	}

	return data[idx];
}

}; //End of namesapce TBTK

#endif
